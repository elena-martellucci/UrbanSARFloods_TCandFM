"""Plot both dates and their difference beside the ground-truth mask for one tile, for intensity or coherence."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

from temporal_features import (CO_COHERENCE_VH, CO_COHERENCE_VV, DIFFERENCE_VH, DIFFERENCE_VV, POST_INTENSITY_VH,
                               POST_INTENSITY_VV, PRE_COHERENCE_VH, PRE_COHERENCE_VV, PRE_INTENSITY_VH,
                               PRE_INTENSITY_VV, add_coherence_differences, add_intensity_differences)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("sar", help="Path to the SAR GeoTIFF")
parser.add_argument("mask", help="Path to the matching ground-truth GeoTIFF")
parser.add_argument("--coherence", action="store_true", help="Plot coherence instead of intensity")
parser.add_argument("--output", help="Image to write (default: results/temporal/<tile>.png)")
args = parser.parse_args()

with rasterio.open(args.sar) as sar, rasterio.open(args.mask) as mask:
    image = sar.read(out_dtype="float32")
    labels = mask.read(1)

class_names = ("Non-flooded", "Flooded open", "Flooded urban")
class_colours = ListedColormap(["#e1e0d9", "#2a78d6", "#eb6834"])

if args.coherence:
    image = add_coherence_differences(image)
    quantity, change, second_date, limit = "coherence", "drop, pre minus co", "Co-event", 0.6
    rows = (
        ("VV", PRE_COHERENCE_VV, CO_COHERENCE_VV, DIFFERENCE_VV, 0, 1),
        ("VH", PRE_COHERENCE_VH, CO_COHERENCE_VH, DIFFERENCE_VH, 0, 1),
    )
else:
    image = add_intensity_differences(image)
    quantity, change, second_date, limit = "intensity (dB)", "change, post minus pre (dB)", "Post-event", 10
    rows = (
        ("VV", PRE_INTENSITY_VV, POST_INTENSITY_VV, DIFFERENCE_VV, -25, 0),
        ("VH", PRE_INTENSITY_VH, POST_INTENSITY_VH, DIFFERENCE_VH, -30, -5),
    )

figure, axes = plt.subplots(2, 4, figsize=(14, 9), constrained_layout=True)
for row, (name, pre, post, difference, low, high) in enumerate(rows):
    before = axes[row, 0].imshow(image[pre], cmap="gray", vmin=low, vmax=high)
    axes[row, 1].imshow(image[post], cmap="gray", vmin=low, vmax=high)
    changed = axes[row, 2].imshow(image[difference], cmap="RdBu_r", vmin=-limit, vmax=limit)
    classes = axes[row, 3].imshow(labels, cmap=class_colours, vmin=-0.5, vmax=2.5, interpolation="nearest")

    figure.colorbar(before, ax=axes[row, :2], location="bottom", shrink=0.5, label=f"{name} {quantity}")
    figure.colorbar(changed, ax=axes[row, 2], location="bottom", label=f"{name} {change}")
    legend = figure.colorbar(classes, ax=axes[row, 3], location="bottom", ticks=[0, 1, 2])
    legend.ax.set_xticklabels(class_names)
    axes[row, 0].set_ylabel(name, fontsize=14)

for axis, title in zip(axes[0], ("Pre-event", second_date, "Difference", "Ground truth")):
    axis.set_title(title)
for axis in axes.flat:
    axis.set_xticks([])
    axis.set_yticks([])

tile = Path(args.sar).name.removesuffix("_SAR.tif")
figure.suptitle(tile)
output = Path(args.output or f"results/temporal/{tile}{'_coherence' if args.coherence else ''}.png")
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=100)
print("Saved", output)

print(f"Mean {change}:")
for label, name in enumerate(class_names):
    pixels = (labels == label) & np.isfinite(image[DIFFERENCE_VV]) & np.isfinite(image[DIFFERENCE_VH])
    if pixels.any():
        print(f"{name}: {pixels.sum()} px, VV {image[DIFFERENCE_VV][pixels].mean():+.2f},"
              f" VH {image[DIFFERENCE_VH][pixels].mean():+.2f}")
