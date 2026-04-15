import sys
import argparse
from pathlib import Path
import rasterio
import rioxarray as rxr
import pickle
import numpy as np
import torch
from rasterio.enums import Resampling

#use this to make sure model.py is importable if it's not in the same directory as this script
#sys.path.insert(0, str(Path(__file__).parent.parent))  # add parent directory to path, or whatever path is needed to find model.py
from model import EWTModel

# constants — change based on your setup / model iteration / data files
N_FEATURES = 9
NODATA = -9999.0
#REPO_ROOT = Path("<your project directory here>")


'''
Command-line arguments for batch processing multiple HLS tiles. Adjust paths and parameters as needed:
--hls: Directory containing input HLS tiles (.tif). Expects HLS files with bands in order: [Blue, Green, Red, NIR, SWIR1, SWIR2]
--output_dir: Directory to save predicted CWC GeoTIFFs
--covar_dir: Directory containing covariate rasters (elev, slope, aspect)
--model: Path to trained PyTorch model (.pt)
--norm_stats: Path to normalization stats pickle file (.pkl)
--batch_size: Number of pixels to process per inference batch (adjust based on GPU memory)

example usage:
python scripts/run_inference.py \
    --hls /path/to/hls_tiles/ \
    --output_dir /path/to/output_cwc/ \
    --covar_dir /path/to/covariates/ \
    --model /path/to/trained_model.pt \ 
    --norm_stats /path/to/norm_stats.pkl
'''
def parse_args():
    p = argparse.ArgumentParser(description="Run inference on all HLS tiles in a directory and export predicted CWC as GeoTIFFs.")
    p.add_argument("--hls",        required=True,  help="Path to directory of input HLS tiles (.tif)")
    p.add_argument("--output_dir", required=True,  help="Directory to save predicted CWC GeoTIFFs")
    p.add_argument("--covar_dir",  required=True,  help="Directory containing covariate rasters (elev, slope, aspect)")
    p.add_argument("--model",      required=True,  help="Path to trained PyTorch model (.pt)")
    p.add_argument("--norm_stats", required=True,  help="Path to normalization stats pickle file (.pkl)")
    p.add_argument("--batch_size", type=int, default=4096, help="Pixels per inference batch")
    return p.parse_args()


# mask low-NDVI pixels (non-vegetation) in HLS tiles — adjust threshold as needed
def hls_NDVI_mask(hls_arr, NDVI_THRESHOLD=0.2):
    # HLS bands: [Blue, Green, Red, NIR, SWIR1, SWIR2]
    red = hls_arr[2]
    nir = hls_arr[3]
    denom = nir + red
    denom = np.where(denom != 0, denom, np.nan)
    ndvi = (nir - red) / denom
    return ndvi <= NDVI_THRESHOLD


def is_valid(arr, nodata=NODATA):
    return (arr != nodata) & ~np.isnan(arr) & ~np.isinf(arr)


def _find_covar_file(covar_dir, keyword):
    """Return the first .tif in covar_dir whose name contains keyword (case-insensitive), or raise."""
    covar_path = Path(covar_dir)
    matches = sorted(f for f in covar_path.glob("*.tif") if keyword.lower() in f.name.lower())
    if not matches:
        raise FileNotFoundError(f"No .tif file containing '{keyword}' found in {covar_dir}")
    return matches[0]


'''
Main processing function for a single HLS file.
- Resamples HLS to 60m in native HLS CRS
- Reprojects slope/aspect/elevation to match HLS 60m grid
- Runs model inference and saves predicted CWC as GeoTIFF
'''
def process_file(hls_path, args, stats, model, device, output_dir, suffix):
    print(f"\n{'=' * 60}")
    print(f"Processing: {hls_path.name}")
    print(f"{'=' * 60}")

    output_path = output_dir / f"{hls_path.stem}{suffix}.tif"

    ##resample HLS to 60m in native CRS
    print(f"\n[1/6] Resampling HLS to 60m (native CRS)...")
    hls = rxr.open_rasterio(hls_path)
    if hls.rio.crs.is_geographic:
        # geographic CRS (degrees): ~60m in degrees at equator
        target_res = 60 / 111320
    else:
        # projected CRS (meters)
        target_res = 60
    hls_60m = hls.rio.reproject(hls.rio.crs, resolution=target_res, resampling=Resampling.average)
    del hls

    hls_arr   = hls_60m.values.astype(np.float32)  # (bands, H, W)
    transform = hls_60m.rio.transform()
    crs       = hls_60m.rio.crs
    H, W      = hls_60m.shape[-2], hls_60m.shape[-1]
    print(f"      Shape: ({hls_arr.shape[0]}, {H}, {W})  |  CRS: {crs}")

    #reproject covariates to match HLS 60m grid
    print(f"\n[2/6] Reprojecting covariates to HLS 60m grid...")
    covar_patterns = [("elev", "elevation"), ("slope", "slope"), ("aspect", "aspect")]
    covar_bands = []
    for glob_pat, name in covar_patterns:
        cov_path = _find_covar_file(args.covar_dir, glob_pat)
        print(f"      {name}: {cov_path.name}")
        cov = rxr.open_rasterio(cov_path, masked=True) 
        cov_matched = cov.rio.reproject_match(hls_60m)
        covar_bands.append(cov_matched.values[0].astype(np.float32))  # squeeze band dim

    covar_arr   = np.stack(covar_bands, axis=0)  # (3, H, W)
    covar_arr[2] = np.sin(covar_arr[2] * np.pi / 180)  # convert aspect (degrees) to sin
    covar_valid = ~np.isnan(covar_arr).any(axis=0)
    del hls_60m
    print(f"Covariate shape: {covar_arr.shape}  |  Valid pixels: {covar_valid.sum():,} / {H * W:,}")

    #mask invalid HLS pixels (e.g. nodata, low NDVI) — adjust thresholds as needed
    hls_arr[hls_arr < -1] = np.nan
    ndvi_mask = hls_NDVI_mask(hls_arr, NDVI_THRESHOLD=0.2)
    hls_arr[:, ndvi_mask] = np.nan
    hls_valid = ~np.isnan(hls_arr).any(axis=0)  # (H, W)
    print(f"\n[3/6] HLS valid pixels: {hls_valid.sum():,} / {H * W:,}")

    
    #mask out combined valid pixels (in both HLS and covariates)
    combined_valid = hls_valid & covar_valid
    n_valid = combined_valid.sum()
    print(f"\n[4/6] Combined valid pixels: {n_valid:,} / {H * W:,}")

    if n_valid == 0:
        print("      No valid pixels after masking — skipping file.")
        return

    #normalize inputs using pre-computed stats (mean/std) from training data
    print(f"\n[5/6] Normalizing inputs...")
    hls_mean  = stats["hls"]["mean"].astype(np.float32)
    hls_std   = stats["hls"]["std"].astype(np.float32)
    cov_mean  = stats["covar"]["mean"].astype(np.float32)
    cov_std   = stats["covar"]["std"].astype(np.float32)
    emit_mean = float(stats["emit"]["mean"])
    emit_std  = float(stats["emit"]["std"])

    hls_norm = np.where(
        hls_valid[None, :, :],
        (hls_arr - hls_mean[:, None, None]) / hls_std[:, None, None],
        0.0
    ).astype(np.float32)

    covar_norm = np.where(
        covar_valid[None, :, :],
        (covar_arr - cov_mean[:, None, None]) / cov_std[:, None, None],
        0.0
    ).astype(np.float32)

    #stack features and run inference
    print(f"\n[6/6] Stacking features and running inference (batch_size={args.batch_size})...")
    features     = np.concatenate([hls_norm, covar_norm], axis=0)  # (N_FEATURES, H, W)
    pixels_flat  = features.reshape(N_FEATURES, -1).T
    valid_pixels = pixels_flat[combined_valid.ravel()]

    all_preds = []
    n_batches = (n_valid + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for i in range(0, n_valid, args.batch_size):
            batch = torch.from_numpy(valid_pixels[i : i + args.batch_size]).float().to(device)
            all_preds.append(model(batch).cpu().numpy())
            if (i // args.batch_size + 1) % max(1, n_batches // 10) == 0:
                print(f"      Batch {i // args.batch_size + 1}/{n_batches}", end="\r")

    preds_valid = np.concatenate(all_preds)
    print(f"      Done — {len(preds_valid):,} predictions")

    #denormalize to get real cwc
    # mask out of range values is optional 
    # (right now we do not, but can bound on (0, max CWC from isofit)
    denorm = preds_valid * emit_std + emit_mean
    EWT_MIN, EWT_MAX = -100.0, 100.0
    ood_mask = (denorm < EWT_MIN) | (denorm > EWT_MAX)
    if ood_mask.sum() > 0:
        print(f"      Masking {ood_mask.sum():,} OOD predictions")
        denorm[ood_mask] = NODATA

    output_arr = np.full(H * W, NODATA, dtype=np.float32)
    output_arr[combined_valid.ravel()] = denorm
    output_arr = output_arr.reshape(H, W)

    final_valid = output_arr != NODATA
    if final_valid.any():
        print(f"\n      Predicted CWC range: [{output_arr[final_valid].min():.4f}, {output_arr[final_valid].max():.4f}]")
        print(f"      Mean predicted CWC:  {output_arr[final_valid].mean():.4f}")
        print(f"      Valid output pixels: {final_valid.sum():,}")

    #save prediction geotiff
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=NODATA,
        compress="lzw",
    ) as dst:
        dst.write(output_arr, 1)

    print(f"\n      Saved: {output_path}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hls_files = sorted(Path(args.hls).glob("*.tif"))
    if not hls_files:
        print(f"No .tif files found in {args.hls}")
        sys.exit(1)
    print(f"Found {len(hls_files)} HLS tile(s) to process.")

    # load shared resources once
    print("\nLoading model and normalization stats...")
    with open(args.norm_stats, "rb") as f:
        stats = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = EWTModel(n_features=N_FEATURES).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    for hls_path in hls_files:
        process_file(hls_path, args, stats, model, device, output_dir, "_predicted_cwc")

    print(f"\nDone. {len(hls_files)} file(s) processed → {output_dir}")


if __name__ == "__main__":
    main()
