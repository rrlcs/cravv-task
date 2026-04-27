"""Cooking-state classification dataset.

The directory layout is:
    <root>/classification/images/<state>/<filename>.jpg
where <state> in {"0", "0.25", "0.50", "0.75", "1.0"}.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import letterbox, normalize, to_tensor


# Sorted by numeric "doneness" so index 0 == raw, index 4 == fully cooked.
STATE_LABELS: Tuple[str, ...] = ("0", "0.25", "0.50", "0.75", "1.0")
LABEL_TO_INDEX = {name: i for i, name in enumerate(STATE_LABELS)}
INDEX_TO_LABEL = {i: name for i, name in enumerate(STATE_LABELS)}


class ClassificationDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, indices: List[int] = None) -> None:
        self.image_size = image_size
        self.root = Path(root) / "classification" / "images"

        self.samples: List[Tuple[Path, int]] = []
        for state in STATE_LABELS:
            folder = self.root / state
            if not folder.exists():
                continue
            label = LABEL_TO_INDEX[state]
            for p in sorted(folder.glob("*.jpg")):
                self.samples.append((p, label))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img, _, _, _ = letterbox(img, self.image_size)
        img_t = normalize(to_tensor(img))
        return img_t, torch.tensor(label, dtype=torch.long)
