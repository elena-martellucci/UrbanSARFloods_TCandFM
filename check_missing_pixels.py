"""Count missing SAR pixels in both splits and list the tiles without any valid pixel."""

import argparse
from pathlib import Path

import numpy as np
import rasterio

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", default="data/urban_sar_floods")
parser.add_argument("--output", default="empty_tiles.txt", help="File to write the tiles without any valid pixel to")
args = parser.parse_args()

root = Path(args.data_root)
names = ("Non-flooded", "Flooded open", "Flooded urban")
empty_tiles = []

for split, list_name in (("Training", "Train_dataset.txt"), ("Validation", "Valid_dataset.txt")):
    lines = (root / list_name).read_text().split()
    tiles_with_missing = 0
    tiles_empty = 0
    pixels = np.zeros(3, dtype=np.int64)
    pixels_missing = np.zeros(3, dtype=np.int64)
    pixels_in_tiles_with_missing = np.zeros(3, dtype=np.int64)

    for line in lines:
        mask_path = root / line.removeprefix("../")
        sar_path = mask_path.parent.parent / "SAR" / mask_path.name.replace("_GT.tif", "_SAR.tif")
        with rasterio.open(sar_path) as sar, rasterio.open(mask_path) as mask:
            missing = ~np.isfinite(sar.read()).all(axis=0)
            labels = mask.read(1).astype(np.int64)

        counts = np.bincount(labels.ravel(), minlength=3)
        pixels += counts
        if missing.any():
            tiles_with_missing += 1
            pixels_in_tiles_with_missing += counts
            pixels_missing += np.bincount(labels[missing], minlength=3)
        if missing.all():
            tiles_empty += 1
            empty_tiles.append(line)

    print(f"{split}: {len(lines)} tiles, {tiles_with_missing} with missing pixels"
          f" ({tiles_with_missing / len(lines):.1%}), {tiles_empty} without any valid pixel")
    print(f"Missing pixels: {pixels_missing.sum() / pixels.sum():.1%} of all pixels")
    print("Class               Pixels      Missing          In tiles with missing pixels")
    for name, total, gone, inside in zip(names, pixels, pixels_missing, pixels_in_tiles_with_missing):
        print(f"{name:<14} {total:11d} {gone:12d} {gone / total:6.1%} {inside:21d} {inside / total:6.1%}")
    print()

Path(args.output).write_text("".join(line + "\n" for line in empty_tiles))
print(f"Wrote {len(empty_tiles)} tiles without any valid pixel to {args.output}")
