# CWC Upscaling

A neural network model for predicting canopy water content (CWC / EWT) from Harmonized Landsat Sentinel (HLS) satellite imagery and topographic covariates.

## Overview

The model takes HLS multispectral tiles and terrain data as input and produces spatially explicit CWC predictions at 60m resolution as GeoTIFFs. Low-NDVI pixels (non-vegetated areas) are masked before inference. Current iteration is trained only on dry season Southern Sierra ROI. Generalize with caution!

**Inputs:**
- HLS spectral bands: Blue, Green, Red, NIR, SWIR1, SWIR2
- Topographic covariates: elevation, slope, aspect (sin-transformed)

**Output:** Predicted CWC raster (GeoTIFF, 60m resolution)

## Usage

### Run Inference

```bash
python scripts/run_inference.py \
    --hls /path/to/hls_tiles/ \
    --output_dir /path/to/output_cwc/ \
    --covar_dir /path/to/covariates/ \
    --model trained_models/ewt_model.pt \
    --norm_stats trained_models/norm_stats.pkl \
    --batch_size 4096
```

**Arguments:**

| Argument | Description |
|---|---|
| `--hls` | Directory of input HLS tiles (`.tif`) |
| `--output_dir` | Directory to save predicted CWC GeoTIFFs |
| `--covar_dir` | Directory containing elevation, slope, and aspect rasters |
| `--model` | Path to trained model weights (`.pt`) |
| `--norm_stats` | Path to normalization stats (`.pkl`) |
| `--batch_size` | Pixels per inference batch (default: 4096) |

The covariate directory should contain `.tif` files with filenames containing `elev`, `slope`, and `aspect`.

### Inspect a GeoTIFF
tif_info is a useful little script for quickly inspecting .tif metadata that I've been using

```bash
python scripts/tif_info.py /path/to/file.tif
```

## Dependencies

- Python 3.x
- PyTorch
- rasterio
- rioxarray
- numpy
