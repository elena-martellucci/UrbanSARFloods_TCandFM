"""Check the UrbanSARFloods band order against the ground-truth masks."""

import argparse
from pathlib import Path

import numpy as np
import rasterio

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", default="data/urban_sar_floods")
parser.add_argument("--tiles", type=int, default=100, help="Number of flooded-urban training tiles to read")
args = parser.parse_args()

root = Path(args.data_root)
validation = set((root / "Valid_dataset.txt").read_text().split())
all_masks = [path for path in sorted((root / "03_FU" / "GT").glob("*_GT.tif"))
             if f"../03_FU/GT/{path.name}" not in validation]
if not all_masks:
    parser.error(f"No ground-truth tiles found in {root}/03_FU/GT")
step = max(1, len(all_masks) // args.tiles)
mask_paths = all_masks[::step][:args.tiles]  

names = ("non-flooded", "flooded open", "flooded urban")
band_sums = np.zeros((3, 8))  
pixel_counts = np.zeros(3, dtype=np.int64)
lowest = np.full(8, np.inf)
highest = np.full(8, -np.inf)
urban_rule_pixels = 0

for mask_path in mask_paths:
    sar_path = mask_path.parent.parent / "SAR" / mask_path.name.replace("_GT.tif", "_SAR.tif")
    with rasterio.open(sar_path) as sar, rasterio.open(mask_path) as mask:
        image = sar.read().astype(np.float64)
        labels = mask.read(1)

    valid = np.isfinite(image).all(axis=0)  
    if not valid.any():
        continue
    lowest = np.minimum(lowest, image[:, valid].min(axis=1))
    highest = np.maximum(highest, image[:, valid].max(axis=1))
    for label in range(3):
        pixels = image[:, valid & (labels == label)]
        band_sums[label] += pixels.sum(axis=1)
        pixel_counts[label] += pixels.shape[1]

    urban = image[:, valid & (labels == 2)]
    urban_rule_pixels += int(((urban[[0, 1]] - urban[[2, 3]]) > 0.3).any(axis=0).sum())

means = band_sums / pixel_counts[:, np.newaxis]
non_flooded, flooded_open, flooded_urban = means

print(f"Training tiles read: {len(mask_paths)} of {len(all_masks)} in 03_FU")
print("Valid pixels [non-flooded, flooded open, flooded urban]:", pixel_counts.tolist())
print()
print("Band   Lowest  Highest   Mean non-flooded   Mean flooded open   Mean flooded urban")
for band in range(8):
    print(f"{band + 1:4d} {lowest[band]:8.2f} {highest[band]:8.2f} {non_flooded[band]:18.3f}"
          f" {flooded_open[band]:19.3f} {flooded_urban[band]:20.3f}")
print()
print(f"1. Coherence or intensity? Bands 1-4 lie in [{lowest[:4].min():.2f}, {highest[:4].max():.2f}],"
      f" bands 5-8 in [{lowest[4:].min():.1f}, {highest[4:].max():.1f}] dB.")
print("2. VH or VV? VV is the brighter one. Non-flooded: band 6 - band 5 ="
      f" {non_flooded[5] - non_flooded[4]:+.1f} dB, band 8 - band 7 = {non_flooded[7] - non_flooded[6]:+.1f} dB.")
print("3. Pre- or post-event intensity? Open water is dark. Flooded open: band 7 - band 5 ="
      f" {flooded_open[6] - flooded_open[4]:+.1f} dB, band 8 - band 6 = {flooded_open[7] - flooded_open[5]:+.1f} dB.")
print("4. Pre- or co-event coherence? Flooding lowers it. Flooded urban: band 1 - band 3 ="
      f" {flooded_urban[0] - flooded_urban[2]:+.2f}, band 2 - band 4 = {flooded_urban[1] - flooded_urban[3]:+.2f};"
      f" {urban_rule_pixels / pixel_counts[2]:.1%} of flooded-urban pixels exceed 0.3 in at least one.")
print()
print("Mean change by class      VV intensity   VH intensity   VV coherence drop   VH coherence drop")
for name, mean in zip(names, means):
    print(f"{name:<22} {mean[7] - mean[5]:+11.2f} dB {mean[6] - mean[4]:+11.2f} dB"
          f" {mean[1] - mean[3]:+19.3f} {mean[0] - mean[2]:+19.3f}")
