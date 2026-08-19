# CWC Upscaling

A neural network model for predicting canopy water content (CWC / EWT) from Harmonized Landsat Sentinel (HLS) satellite imagery and topographic covariates.

## Overview

The model takes HLS multispectral tiles and a DEM as input and produces spatially explicit CWC predictions as GeoTIFFs. Slope and aspect are derived from the DEM automatically with `gdaldem`, so no other terrain rasters need to be supplied. Low-NDVI pixels (non-vegetated areas) are masked before inference.

There are some example prediction `.tif` files in [sample_predictions/](sample_predictions/).

## Trained models

Two models ship with the repo. They were trained on different areas and different covariate sets, so each has its own architecture, feature count, and normalization stats — the `--full_state` flag selects between them and everything else follows from it.

| | default (Sierra ROI) | `--full_state` (statewide) |
|---|---|---|
| Weights | `trained_models/sierra/ewt_model.pt` | `trained_models/ca_state/ca_state_ewt_model.pt` |
| Norm stats | `trained_models/sierra/norm_stats.pkl` | `trained_models/ca_state/norm_stats.pkl` |
| Definition | [scripts/sierra_model.py](scripts/sierra_model.py) | [scripts/full_state_model.py](scripts/full_state_model.py) |
| Features | 9 | 10 |
| Hidden dims | `[64, 32, 16]` | `[128, 128, 128, 128]` |
| Aspect encoding | single `sin(aspect)` band | `northing = cos` + `easting = sin` |
| Training area | dry season Southern Sierra ROI | California statewide, year-round |

**Inputs:**
- HLS spectral bands, in this order: Blue, Green, Red, NIR, SWIR1, SWIR2
- Topographic covariates, derived from the supplied DEM: elevation, slope, aspect

**Output:** Predicted CWC raster (GeoTIFF), at either 30m or, pick whichever suits your application with `--30m` / `--60m`. Nodata is `-9999.0`.

The Sierra model is trained only on the dry season Southern Sierra ROI — generalize with caution. The statewide model is probably better to use in nearly all cases. 

## Usage

### Run Inference

> **NOTE:** The inference script takes a single DEM file and generates slope/aspect model inputs. Make sure you have `gdaldem` installed and it is in your PATH.
>
> ```bash
> conda install -c conda-forge gdal
> ```
>
> If `gdaldem` is installed but not found, add its location to your PATH, e.g.:
>
> ```bash
> export PATH="/path/to/conda/envs/your_env/bin:$PATH"
> ```

The bundled weights and stats are found automatically, so the common case only needs the three data paths:

```bash
# Sierra ROI model
python scripts/run_inference.py \
    --hls /path/to/hls_tiles/ \
    --output_dir /path/to/output_cwc/ \
    --dem /path/to/dem.tif
```

```bash
# statewide model, at 30m
python scripts/run_inference.py \
    --hls /path/to/hls_tiles/ \
    --output_dir /path/to/output_cwc/ \
    --dem /path/to/dem.tif \
    --full_state --30m
```

```bash
# override the bundled weights with your own
python scripts/run_inference.py \
    --hls /path/to/hls_tiles/ \
    --output_dir /path/to/output_cwc/ \
    --dem /path/to/dem.tif \
    --model /path/to/trained_model.pt \
    --norm_stats /path/to/norm_stats.pkl \
    --batch_size 4096
```

**Arguments:**

| Argument | Description |
|---|---|
| `--hls` | Directory of input HLS tiles (`.tif`) — **required** |
| `--output_dir` | Directory to save predicted CWC GeoTIFFs — **required** |
| `--dem` | DEM (`.tif`) covering the HLS tiles; slope and aspect are derived from it — **required** |
| `--model` | Path to trained model weights (`.pt`). Defaults to the selected region's bundled weights |
| `--norm_stats` | Path to normalization stats (`.pkl`). Defaults to the selected region's bundled stats |
| `--full_state` | Use the statewide model (`trained_models/ca_state/`) instead of the Sierra ROI model |
| `--30m` / `--60m` | Resolution to predict at; both are supported and validated. `--60m` (the default) resamples each tile to native EMIT 60m resolution first; `--30m` uses the tile's native grid |
| `--batch_size` | Pixels per inference batch (default: 4096) — tune to your GPU memory |

`--model` and `--norm_stats` are overrides, not requirements. If you pass one, pass the matching other: a model run against another model's normalization stats will produce wrong values, and a mismatched covariate count is rejected at startup.

Every tile in `--hls` is processed with the same DEM-derived terrain, which is computed once up front rather than per tile. Outputs are named `<input_stem>_predicted_cwc.tif`.

### Inspect a GeoTIFF

`tif_info.py` is a useful little script for quickly inspecting `.tif` metadata that I've been using

```bash
python scripts/tif_info.py /path/to/file.tif
```

## Dependencies

- Python 3.x
- PyTorch
- rasterio
- rioxarray
- numpy
- GDAL — `gdaldem` must be on your `PATH` (e.g. `conda install -c conda-forge gdal`)
