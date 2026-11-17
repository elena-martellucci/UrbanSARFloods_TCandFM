from pathlib import Path

import numpy as np
import rasterio


class UrbanSARFloods:
    def __init__(self, root, split):
        split_files = {
            "train": "Train_dataset.txt",
            "val": "Valid_dataset.txt",
        }
        if split not in split_files:
            raise ValueError("split must be 'train' or 'val'")

        self.root = Path(root)
        split_path = self.root / split_files[split]
        self.mask_paths = [
            self.root / line.strip().removeprefix("../")
            for line in split_path.read_text().splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, index):
        mask_path = self.mask_paths[index]
        tile_id = mask_path.name.removesuffix("_GT.tif")
        sar_path = mask_path.parent.parent / "SAR" / f"{tile_id}_SAR.tif"

        with rasterio.open(sar_path) as sar, rasterio.open(mask_path) as mask:
            aligned = (
                sar.shape == mask.shape
                and sar.crs == mask.crs
                and sar.transform.almost_equals(mask.transform)
            )
            if not aligned or sar.count != 8 or mask.count != 1:
                raise ValueError(f"Unexpected image or mask layout for {tile_id}")

            image = sar.read(out_dtype="float32", masked=True)
            labels = mask.read(1, masked=True)

        if np.ma.getmaskarray(image).any() or not np.isfinite(image.data).all():
            raise ValueError(f"Invalid SAR pixels in {sar_path}")
        if np.ma.getmaskarray(labels).any() or not np.isin(labels.data, [0, 1, 2]).all():
            raise ValueError(f"Invalid labels in {mask_path}")

        return image.data, labels.data.astype(np.int64)