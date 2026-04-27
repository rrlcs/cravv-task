"""Object-detection dataset reading the COCO-format annotations file.

The COCO file lists 4 categories: id 0 ("objects", which is the dummy
super-category), 1 (pan), 2 (stirrer), 3 (tap). We drop category id 0 and
remap remaining ids contiguously to [0, 1, 2].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import letterbox, normalize, to_tensor


# Final, model-side class indices (after dropping the parent "objects").
DET_CLASS_NAMES: Tuple[str, ...] = ("pan", "stirrer", "tap")


class DetectionDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, indices: List[int] = None) -> None:
        self.image_size = image_size
        det_root = Path(root) / "detection"
        self.images_dir = det_root / "images"
        with open(det_root / "_annotations.coco.json") as f:
            coco = json.load(f)

        # Build {coco_cat_id: model_class_index}, dropping the dummy parent.
        coco_cats = coco["categories"]
        keep_names = ("pan", "stirrer", "tap")
        cat_id_to_idx = {}
        for c in coco_cats:
            if c["name"] in keep_names:
                cat_id_to_idx[c["id"]] = keep_names.index(c["name"])
        self.cat_id_to_idx = cat_id_to_idx

        # Group annotations by image id.
        anns_by_img = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        self.samples = []
        for img_info in coco["images"]:
            img_id = img_info["id"]
            file_path = self.images_dir / img_info["file_name"]
            if not file_path.exists():
                continue
            anns = anns_by_img.get(img_id, [])
            self.samples.append((file_path, anns))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, anns = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        boxes_xyxy = []
        labels = []
        for a in anns:
            cat = a["category_id"]
            if cat not in self.cat_id_to_idx:
                continue  # skip the dummy parent category
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[cat])

        boxes = torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.zeros((0, 4))
        labels_t = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        img, _, boxes, _ = letterbox(img, self.image_size, boxes=boxes)
        img_t = normalize(to_tensor(img))
        if boxes is None:
            boxes = torch.zeros((0, 4))

        target = {"boxes": boxes, "labels": labels_t}
        return img_t, target


def det_collate_fn(batch):
    """Stack images, but keep targets as a list (variable number of boxes per image)."""
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets
