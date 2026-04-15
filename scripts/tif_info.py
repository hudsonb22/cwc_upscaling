#!/usr/bin/env python3
"""Print metadata and stats for a GeoTIFF file."""

import sys
import numpy as np
import rasterio


def main():
    if len(sys.argv) < 2:
        print("Usage: python tif_info.py <path_to_tif>")
        sys.exit(1)

    filepath = sys.argv[1]

    with rasterio.open(filepath) as src:
        data = src.read()

        print(f"File: {filepath}")
        print("-" * 60)

        # Shape and type
        print(f"Shape (bands, height, width): {data.shape}")
        print(f"Dtype: {data.dtype}")

        # Spatial info
        print(f"\nCRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Resolution (x, y): ({src.res[0]:.6f}, {src.res[1]:.6f})")
        print(f"Transform:\n{src.transform}")

        # NoData
        print(f"\nNoData value: {src.nodata}")


        # Stats per band
        print("\nBand Statistics:")
        for i in range(data.shape[0]):
            band = data[i]
            if src.nodata is not None and not np.isnan(src.nodata):
                valid = band[(band != src.nodata) & ~np.isnan(band)]
            else:
                valid = band[~np.isnan(band)]
            if valid.size > 0:
                print(f"  Band {i+1}: min={valid.min():.4f}, max={valid.max():.4f}, "
                      f"mean={valid.mean():.4f}, std={valid.std():.4f}")
                nodata_pct = (1 - valid.size / band.size) * 100
                print(f"           NoData pixels: {nodata_pct:.1f}%")
            else:
                print(f"  Band {i+1}: all NoData")


        


if __name__ == "__main__":
    main()
