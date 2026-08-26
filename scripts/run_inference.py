import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
import rasterio
import rioxarray as rxr
import pickle
import numpy as np
import torch
from rasterio.enums import Resampling

#the two model definitions live next to this script; add the script's own directory to
#sys.path so the imports below work no matter where this is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sierra_model import EWTModel as SierraEWTModel          # Sierra ROI model  (LeakyReLU)
from full_state_model import EWTModel as FullStateEWTModel   # statewide model   (Softplus)

# constants — change based on your setup / model iteration / data files
NODATA = -9999.0
REPO_ROOT = Path(__file__).resolve().parent.parent

# default (Sierra ROI) model — trained on the cropped Southern Sierra area, on 9 features:
# [Blue, Green, Red, NIR, SWIR1, SWIR2, elevation, slope, aspect_sin]
SIERRA_DIR         = REPO_ROOT / "trained_models" / "sierra"
SIERRA_MODEL_PATH  = SIERRA_DIR / "ewt_model.pt"
SIERRA_NORM_STATS  = SIERRA_DIR / "norm_stats.pkl"
N_FEATURES         = 9
HIDDEN_DIMS        = [64, 32, 16]

# full-state (statewide) model — selected with --full_state.
# It was trained without the HAND covariate, on 10 features:
# [Blue, Green, Red, NIR, SWIR1, SWIR2, elevation, slope, aspect_northing, aspect_easting]
# Aspect is decomposed into northing=cos / easting=sin rather than the single
# sin band the default model uses, so both the feature count and the covariate
# construction change with the flag.
FULL_STATE_DIR         = REPO_ROOT / "trained_models" / "ca_state"
FULL_STATE_MODEL_PATH  = FULL_STATE_DIR / "ca_state_ewt_model.pt"
FULL_STATE_NORM_STATS  = FULL_STATE_DIR / "norm_stats.pkl"
FULL_STATE_N_FEATURES  = 10
FULL_STATE_HIDDEN_DIMS = [128, 128, 128, 128]


'''
Command-line arguments for batch processing multiple HLS tiles. Adjust paths and parameters as needed:
--hls: Directory containing input HLS tiles (.tif). Expects HLS files with bands in order: [Blue, Green, Red, NIR, SWIR1, SWIR2]
--output_dir: Directory to save predicted CWC GeoTIFFs
--dem: Path to a DEM (.tif) covering the HLS tiles. Slope and aspect are derived from it
    with gdaldem, so no other terrain rasters need to be supplied.
--model: Path to trained PyTorch model (.pt). Defaults to the model for the selected
    region — trained_models/sierra/ewt_model.pt, or trained_models/ca_state/ca_state_ewt_model.pt
    with --full_state — so it only needs to be passed to override those.
--norm_stats: Path to normalization stats pickle file (.pkl). Defaults, like --model, to the
    norm_stats.pkl sitting in the selected region's trained_models/ folder.
--batch_size: Number of pixels to process per inference batch (adjust based on GPU memory)
--full_state: Use the statewide model (trained_models/ca_state/) instead of the default
    Sierra ROI model (trained_models/sierra/). The two have different architectures and
    feature counts, so this flag also selects which EWTModel class is built.
--30m / --60m: Resolution to run at. Both are supported and have been verified to work well,
    so pick whichever suits your application. --60m (the default) resamples each HLS tile to
    60m in its native CRS first; --30m skips that resampling and predicts on the tile's
    native 30m grid.

example usage — Sierra ROI model, using the bundled weights/stats:
python scripts/run_inference.py \
    --hls /path/to/hls_tiles/ \
    --output_dir /path/to/output_cwc/ \
    --dem /path/to/dem.tif

# statewide model, at 30m:
python scripts/run_inference.py \
    --hls /path/to/hls_tiles/ \
    --output_dir /path/to/output_cwc/ \
    --dem /path/to/dem.tif \
    --full_state --30m

# either one with explicitly-supplied weights:
python scripts/run_inference.py \
    --hls /path/to/hls_tiles/ \
    --output_dir /path/to/output_cwc/ \
    --dem /path/to/dem.tif \
    --model /path/to/trained_model.pt \
    --norm_stats /path/to/norm_stats.pkl
'''
def parse_args():
    p = argparse.ArgumentParser(description="Run inference on all HLS tiles in a directory and export predicted CWC as GeoTIFFs.")
    p.add_argument("--hls",        required=True,  help="Path to directory of input HLS tiles (.tif)")
    p.add_argument("--output_dir", required=True,  help="Directory to save predicted CWC GeoTIFFs")
    p.add_argument("--dem",        required=True,  help="Path to a DEM (.tif); slope and aspect are derived from it with gdaldem")
    p.add_argument("--model",      default=None,
                   help="Path to trained PyTorch model (.pt). Defaults to the selected region's "
                        f"bundled weights ({SIERRA_MODEL_PATH.name}, or {FULL_STATE_MODEL_PATH.name} "
                        "with --full_state)")
    p.add_argument("--norm_stats", default=None,
                   help="Path to normalization stats pickle file (.pkl). Defaults to the norm_stats.pkl "
                        "in the selected region's trained_models/ folder")
    p.add_argument("--batch_size", type=int, default=4096, help="Pixels per inference batch")
    p.add_argument("--full_state", action="store_true",
                   help=f"Use the statewide model ({FULL_STATE_DIR.name}/) instead of the default "
                        f"Sierra ROI model ({SIERRA_DIR.name}/)")

    # --30m / --60m both write to args.resolution; 60m is the default
    res = p.add_mutually_exclusive_group()
    res.add_argument("--30m", dest="resolution", action="store_const", const=30,
                     help="Predict on the HLS tile's native 30m grid (skips the 60m resampling)")
    res.add_argument("--60m", dest="resolution", action="store_const", const=60,
                     help="Resample each HLS tile to 60m before predicting (default)")
    p.set_defaults(resolution=60)

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


def decompose_aspect(aspect_deg):
    """Split aspect (degrees clockwise from north) into northing/easting components.

    Aspect is circular — 359° and 1° are neighbors but average to 180° — so it
    can't be fed to the model, or resampled, as a raw angle. The full-state model
    was trained on the two components, in this order:
        northing = cos(aspect)   (+1 = due north, -1 = due south)
        easting  = sin(aspect)   (+1 = due east,  -1 = due west)

    Returns (northing, easting), to be stacked in that order.
    """
    rad = np.deg2rad(aspect_deg.astype(np.float32))
    northing = np.cos(rad).astype(np.float32)
    easting  = np.sin(rad).astype(np.float32)
    return northing, easting


def derive_terrain(dem_path, work_dir):
    """Derive slope and aspect from the DEM with gdaldem, so only a DEM has to be supplied.

    These flags match how the training covariates were generated (`-compute_edges`, no
    explicit scale). Note that for a DEM in a geographic CRS gdaldem still applies its own
    horizontal scaling — xscale = 111320 * cos(center_latitude), yscale = 111320 — derived
    once from the DEM's own center latitude. A differently-cropped DEM therefore yields
    slightly differently-scaled slope than the statewide DEM the training rasters came
    from; see the commented-out scale pinning below if that matters to you.

    Returns {"elevation": ..., "slope": ..., "aspect": ...} paths.
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise SystemExit(f"DEM not found: {dem_path}")

    paths = {"elevation": dem_path}
    for mode in ("slope", "aspect"):
        out_path = Path(work_dir) / f"{mode}.tif"
        print(f"      gdaldem {mode} → {out_path.name}")
        cmd = ["gdaldem", mode, str(dem_path), str(out_path), "-compute_edges"]

        # Optional: pin slope's horizontal scaling to the training DEM's, so derived slope
        # doesn't shift with how the user happened to crop their DEM. 88588.4 is
        # 111320 * cos(37.2692degN), the statewide training DEM's center latitude.
        # Verified to reproduce the training slope raster to within 1.3e-4 degrees. The
        # measured effect on predictions is nil (r = 1.00000 for a ~3% slope shift), so
        # this is about reproducibility across DEM extents, not accuracy. Slope only —
        # gdaldem aspect takes no scale options and is unaffected.
        # if mode == "slope":
        #     cmd += ["-xscale", "88588.4", "-yscale", "111320"]

        try:
            subprocess.run(
                cmd,
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
        except FileNotFoundError:
            raise SystemExit("gdaldem not found on PATH — install GDAL (e.g. `conda install -c conda-forge gdal`)")
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"gdaldem {mode} failed:\n{e.stderr}")
        paths[mode] = out_path
    return paths


'''
Main processing function for a single HLS file.
- Resamples HLS to 60m in native HLS CRS (or keeps the native 30m grid with --30m)
- Reprojects slope/aspect/elevation (from `terrain`, see derive_terrain) to match that HLS grid
- Runs model inference and saves predicted CWC as GeoTIFF
'''
def process_file(hls_path, args, stats, model, device, output_dir, suffix, terrain):
    print(f"\n{'=' * 60}")
    print(f"Processing: {hls_path.name}")
    print(f"{'=' * 60}")

    output_path = output_dir / f"{hls_path.stem}{suffix}.tif"

    ##resample HLS to 60m in native CRS, or keep the native 30m grid
    # keep the source handle separate from hls_ref: reprojecting rebinds hls_ref to a new
    # in-memory object, and the file it came from still has to be closed explicitly (below).
    hls_src = rxr.open_rasterio(hls_path)
    hls_ref = hls_src
    if args.resolution == 30:
        print(f"\n[1/6] Using HLS at native 30m resolution (no resampling)...")
    else:
        print(f"\n[1/6] Resampling HLS to 60m (native CRS)...")
        if hls_src.rio.crs.is_geographic:
            # geographic CRS (degrees): ~60m in degrees at equator
            target_res = 60 / 111320
        else:
            # projected CRS (meters)
            target_res = 60
        hls_ref = hls_src.rio.reproject(hls_src.rio.crs, resolution=target_res, resampling=Resampling.average)

    hls_arr   = hls_ref.values.astype(np.float32)  # (bands, H, W)
    transform = hls_ref.rio.transform()
    crs       = hls_ref.rio.crs
    H, W      = hls_ref.shape[-2], hls_ref.shape[-1]
    print(f"      Shape: ({hls_arr.shape[0]}, {H}, {W})  |  CRS: {crs}")

    #reproject covariates to match the HLS grid
    print(f"\n[2/6] Reprojecting covariates to HLS {args.resolution}m grid...")
    covar_bands = []
    for name in ("elevation", "slope", "aspect"):
        cov_path = terrain[name]
        print(f"      {name}: {cov_path.name}")
        # close each covariate before moving on — slope/aspect live in a temp dir that has
        # to be removable at exit, and Windows refuses to delete a file with an open handle
        with rxr.open_rasterio(cov_path, masked=True) as cov:
            cov_matched = cov.rio.reproject_match(hls_ref)
            covar_bands.append(cov_matched.values[0].astype(np.float32))  # squeeze band dim

    if args.full_state:
        # full-state model expects aspect split into northing/easting components
        elevation, slope = covar_bands[0], covar_bands[1]
        aspect_northing, aspect_easting = decompose_aspect(covar_bands[2])
        # gdaldem flags flat cells (slope 0) with the same nodata value as real gaps.
        # Training gave those a (0, 0) aspect vector instead of dropping them, since a
        # flat cell has no meaningful direction — cross-reference slope to tell them apart.
        flat = (slope == 0.0) & np.isnan(covar_bands[2])
        aspect_northing[flat] = 0.0
        aspect_easting[flat]  = 0.0
        # order must match training: [elevation, slope, aspect_northing, aspect_easting]
        covar_arr = np.stack([elevation, slope, aspect_northing, aspect_easting], axis=0)  # (4, H, W)
    else:
        covar_arr   = np.stack(covar_bands, axis=0)  # (3, H, W)
        covar_arr[2] = np.sin(covar_arr[2] * np.pi / 180)  # convert aspect (degrees) to sin
    covar_valid = ~np.isnan(covar_arr).any(axis=0)
    del hls_ref
    hls_src.close()
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

    # the full-state norm stats carry a trailing HAND entry that this model
    # (trained without HAND) doesn't use — drop it to line up with covar_arr
    n_covar = covar_arr.shape[0]
    if len(cov_mean) == n_covar + 1:
        cov_mean = cov_mean[:n_covar]
        cov_std  = cov_std[:n_covar]
    elif len(cov_mean) != n_covar:
        raise ValueError(
            f"--norm_stats has {len(cov_mean)} covariate entries but {n_covar} covariate bands were "
            f"built. Check that --norm_stats matches the selected model."
        )

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
    features     = np.concatenate([hls_norm, covar_norm], axis=0)  # (n_features, H, W)
    pixels_flat  = features.reshape(features.shape[0], -1).T
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

    # --full_state swaps in the statewide model (trained_models/ca_state), which has its
    # own architecture, feature count and layer sizes; without it we run the Sierra ROI
    # model (trained_models/sierra). --model / --norm_stats override the bundled paths.
    if args.full_state:
        model_path = Path(args.model) if args.model else FULL_STATE_MODEL_PATH
        stats_path = Path(args.norm_stats) if args.norm_stats else FULL_STATE_NORM_STATS
        model = FullStateEWTModel(n_features=FULL_STATE_N_FEATURES,
                                  hidden_dims=FULL_STATE_HIDDEN_DIMS)
    else:
        model_path = Path(args.model) if args.model else SIERRA_MODEL_PATH
        stats_path = Path(args.norm_stats) if args.norm_stats else SIERRA_NORM_STATS
        model = SierraEWTModel(n_features=N_FEATURES, hidden_dims=HIDDEN_DIMS)

    for label, path in (("model", model_path), ("norm stats", stats_path)):
        if not path.exists():
            print(f"{label} not found: {path}")
            sys.exit(1)

    # load shared resources once
    print("\nLoading model and normalization stats...")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = model.to(device)
    print(f"Model: {model_path}  |  norm stats: {stats_path}  |  resolution: {args.resolution}m")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # derive slope/aspect from the DEM once up front, not per tile — gdaldem over a
    # large DEM is slow and the products are identical for every tile. The temp dir
    # is cleaned up on exit; point derive_terrain at a real directory to keep them.
    print(f"\nDeriving terrain covariates from {Path(args.dem).name}...")
    with tempfile.TemporaryDirectory(prefix="terrain_") as work_dir:
        terrain = derive_terrain(args.dem, work_dir)

        for hls_path in hls_files:
            process_file(hls_path, args, stats, model, device, output_dir, "_predicted_cwc", terrain)

    print(f"\nDone. {len(hls_files)} file(s) processed → {output_dir}")


if __name__ == "__main__":
    main()
