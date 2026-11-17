import argparse

import numpy as np
import rasterio

parser = argparse.ArgumentParser()
parser.add_argument("sar", help="Path to the SAR GeoTIFF")
parser.add_argument("mask", help="Path to the matching ground-truth GeoTIFF")
args = parser.parse_args()

with rasterio.open(args.sar) as sar, rasterio.open(args.mask) as mask:
    image = sar.read(masked=True)
    labels = mask.read(1, masked=True)

    same_grid = (
        sar.shape == mask.shape
        and sar.crs == mask.crs
        and sar.transform.almost_equals(mask.transform)
    )

    print("SAR shape (bands, rows, columns):", image.shape)
    print("Mask shape (rows, columns):", labels.shape)
    print("Same grid:", same_grid)
    print("Mask NoData value:", mask.nodata)
    print("Valid label values and pixel counts:", np.unique(
        labels.compressed(), return_counts=True
    ))

    for index, band in enumerate(image, start=1):
        values = band.compressed()
        values = values[np.isfinite(values)]
        if values.size:
            print(f"Band {index}: {values.min():.3f} to {values.max():.3f}")
        else:
            print(f"Band {index}: no valid values")