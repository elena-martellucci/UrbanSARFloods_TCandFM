# UrbanSARFloods — Temporal Change & Foundation Models

Urban Computing | Group 15

## Project overview

We investigate urban flood segmentation using Sentinel-1 SAR imagery. We plan to compare a conventional U-Net and a pretrained SAR-capable foundation model, each with and without explicit temporal-change features. We will evaluate performance on held-out flood events.

The current code loads the official training and validation splits and inspects individual SAR images and their ground-truth masks.

## Setup

Use Python 3.11 or newer. The data loader was tested with Python 3.11.

From the repository root, check your Python version, then create the virtual environment and install the dependencies:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `python3` points to a version older than 3.11, use the command for your newer installation, such as `python3.12`, when creating the environment.

## Dataset

Download `urban_sar_floods.tar.gz` from the [UrbanSARFloods release on Hugging Face](https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1/blob/main/urban_sar_floods.tar.gz). This is the approximately 38 GB training and validation archive.
See the [official dataset README](https://github.com/jie666-6/UrbanSARFloods) for the dataset description and separate test downloads.

Extract the archive so the project has this structure:

```text
data/
└── urban_sar_floods/
    ├── 01_NF/
    ├── 02_FO/
    ├── 03_FU/
    ├── Train_dataset.txt
    └── Valid_dataset.txt
```

Each class folder contains `SAR/` and `GT/` subfolders. Git ignores `data/`, so each collaborator needs a local copy of the dataset.

## Check the loader

Run this from the repository root to load one tile from each split:

```bash
python - <<'PY'
from dataset import UrbanSARFloods

for split in ("train", "val"):
    dataset = UrbanSARFloods("data/urban_sar_floods", split)
    image, mask = dataset[0]
    print(f"{split}: {len(dataset)} tiles")
    print(f"  Image: {image.shape}, {image.dtype}")
    print(f"  Mask:  {mask.shape}, {mask.dtype}")
PY
```

Each sample returns the original eight SAR bands as a `float32` array of shape `(8, 512, 512)` and an `int64` mask of shape `(512, 512)`. Mask labels are `0` for non-flooded, `1` for flooded open area, and `2` for flooded urban area. The loader preserves the original SAR values for computing intensity differences before normalisation.

To inspect a specific image and mask, run `python inspect_tile.py path/to/tile_SAR.tif path/to/tile_GT.tif`.

## Team

- Beyza Celep
- Elena Martellucci