"""Light-weight transforms that work for image+mask and image+boxes.

We keep this self-contained (no albumentations dep) for portability:
- ResizeAndPad letterboxes the image to a fixed square size while keeping
  aspect ratio (mask and boxes are transformed identically).
- Normalize/ToTensor follow ImageNet stats.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """RGB PIL image -> (3, H, W) float tensor in [0, 1]."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def normalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Reverse ``normalize`` for visualization (returns a CPU tensor in [0, 1])."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(3, 1, 1)
    out = t * std + mean
    return out.clamp(0, 1)


def letterbox(
    img: Image.Image,
    target: int,
    pad_value: int = 114,
    mask: Image.Image = None,
    boxes: torch.Tensor = None,
) -> Tuple[Image.Image, Image.Image, torch.Tensor, Dict[str, float]]:
    """Resize so the longer side == target, then pad to target x target.

    Returns:
        new_img, new_mask (or None), new_boxes (or None), info dict with
        ``scale``, ``pad_x``, ``pad_y`` (useful for inverse mapping).
    """
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    pad_w = target - new_w
    pad_h = target - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    new_img = Image.new("RGB", (target, target), (pad_value, pad_value, pad_value))
    new_img.paste(img, (pad_x, pad_y))

    new_mask = None
    if mask is not None:
        # NEAREST so labels stay integer.
        m = mask.resize((new_w, new_h), Image.NEAREST)
        new_mask = Image.new(mask.mode, (target, target), 0)
        new_mask.paste(m, (pad_x, pad_y))

    new_boxes = None
    if boxes is not None and len(boxes) > 0:
        new_boxes = boxes.clone().float()
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]] * scale + pad_x
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]] * scale + pad_y

    info = {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "orig_w": w, "orig_h": h}
    return new_img, new_mask, new_boxes, info
