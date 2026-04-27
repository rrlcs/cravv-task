"""Binary semantic segmentation dataset.

Reads RGB images + binary masks (0 / 255) from
    <root>/segmentation/images/*.jpg
    <root>/segmentation/masks/*.png
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import letterbox, normalize, to_tensor


class SegmentationDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, indices: List[int] = None) -> None:
        self.image_size = image_size
        self.images_dir = Path(root) / "segmentation" / "images"
        self.masks_dir = Path(root) / "segmentation" / "masks"

        all_imgs = sorted(self.images_dir.glob("*.jpg"))
        # Keep only images that have a matching mask.
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in all_imgs:
            mask_path = self.masks_dir / (img_path.stem + ".png")
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img, mask, _, _ = letterbox(img, self.image_size, mask=mask)
        img_t = normalize(to_tensor(img))

        # Binary: foreground == any pixel > 0.
        import numpy as np
        m = np.array(mask, dtype=np.uint8)
        bin_mask = (m > 127).astype("float32")
        mask_t = torch.from_numpy(bin_mask).unsqueeze(0)  # (1, H, W)

        return img_t, mask_t
