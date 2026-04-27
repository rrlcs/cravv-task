"""Anchor generation, IoU matching, and box (de)coding for the detection head.

Conventions:
- Boxes are stored in absolute pixel coordinates as (x1, y1, x2, y2).
- One anchor per (level, location, ratio, scale) combination, flattened into
  a single (A, 4) tensor in the same order the head will produce predictions.
- Box deltas follow the standard Fast R-CNN parameterization:
      tx = (gx - ax) / aw,  ty = (gy - ay) / ah
      tw = log(gw / aw),    th = log(gh / ah)
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------

def _generate_cell_anchors(base_size: float, scales: List[float], ratios: List[float]) -> torch.Tensor:
    """Generate K anchors centered at (0, 0) for one feature map cell.

    Returns: (K, 4) tensor in (x1, y1, x2, y2) at the input image scale.
    """
    anchors = []
    for scale in scales:
        size = base_size * scale
        area = size * size
        for ratio in ratios:
            w = math.sqrt(area / ratio)
            h = w * ratio
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors, dtype=torch.float32)


def generate_anchors_for_level(
    feature_size: Tuple[int, int],
    stride: int,
    cell_anchors: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Tile cell anchors over a feature map.

    Returns: (H * W * K, 4) tensor in image-pixel coordinates.
    """
    H, W = feature_size
    shifts_y = torch.arange(0, H, dtype=torch.float32, device=device) * stride + stride / 2
    shifts_x = torch.arange(0, W, dtype=torch.float32, device=device) * stride + stride / 2
    sy, sx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shifts = torch.stack([sx, sy, sx, sy], dim=-1).reshape(-1, 4)  # (H*W, 4)

    cell = cell_anchors.to(device)
    # (H*W, 1, 4) + (1, K, 4) -> (H*W, K, 4)
    anchors = shifts[:, None, :] + cell[None, :, :]
    return anchors.reshape(-1, 4)


class AnchorGenerator:
    """RetinaNet-style anchor generator.

    For each FPN level we use the same set of (scales, ratios) but different
    base sizes, so anchors grow with stride.
    """

    def __init__(
        self,
        levels: List[str],
        strides: List[int],
        sizes: List[float],  # one base size per level (e.g. 32 for P3)
        ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        scales: Tuple[float, ...] = (1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
    ) -> None:
        assert len(levels) == len(strides) == len(sizes)
        self.levels = levels
        self.strides = strides
        self.sizes = sizes
        self.ratios = list(ratios)
        self.scales = list(scales)
        self.num_anchors_per_loc = len(ratios) * len(scales)

        # Pre-compute per-level cell anchors.
        self._cell_anchors = [
            _generate_cell_anchors(size, self.scales, self.ratios) for size in sizes
        ]

    def __call__(self, feature_sizes: List[Tuple[int, int]], device: torch.device) -> List[torch.Tensor]:
        """Returns a list of (H*W*K, 4) tensors, one per level (same order as ``levels``)."""
        assert len(feature_sizes) == len(self.levels)
        return [
            generate_anchors_for_level(fs, stride, cell, device)
            for fs, stride, cell in zip(feature_sizes, self.strides, self._cell_anchors)
        ]


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU. boxes_a: (N, 4), boxes_b: (M, 4). Returns (N, M)."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)

    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def encode_boxes(anchors: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Encode GT boxes into anchor-relative deltas. Both (N, 4) in xyxy."""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + aw * 0.5
    ay = anchors[:, 1] + ah * 0.5

    gw = (gt[:, 2] - gt[:, 0]).clamp(min=1.0)
    gh = (gt[:, 3] - gt[:, 1]).clamp(min=1.0)
    gx = gt[:, 0] + gw * 0.5
    gy = gt[:, 1] + gh * 0.5

    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Inverse of ``encode_boxes``. Both (N, 4)."""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + aw * 0.5
    ay = anchors[:, 1] + ah * 0.5

    # Clamp dw/dh to avoid exp overflow on raw initialisation.
    dw = deltas[:, 2].clamp(max=4.0)
    dh = deltas[:, 3].clamp(max=4.0)

    gx = deltas[:, 0] * aw + ax
    gy = deltas[:, 1] * ah + ay
    gw = torch.exp(dw) * aw
    gh = torch.exp(dh) * ah

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)


def match_anchors_to_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pos_iou: float = 0.5,
    neg_iou: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each anchor return:
        labels: (N,) int64 -> 0..C-1 for positives, -1 for negatives, -2 for ignored.
        matched_idx: (N,) -> index into gt_boxes (or -1 if no match)

    If gt_boxes is empty, all anchors are negatives.
    """
    n = anchors.shape[0]
    if gt_boxes.numel() == 0:
        labels = torch.full((n,), -1, dtype=torch.long, device=anchors.device)
        matched = torch.full((n,), -1, dtype=torch.long, device=anchors.device)
        return labels, matched

    iou = box_iou(anchors, gt_boxes)  # (N, M)
    max_iou, max_idx = iou.max(dim=1)  # for each anchor

    labels = torch.full((n,), -2, dtype=torch.long, device=anchors.device)  # default = ignore
    labels[max_iou < neg_iou] = -1  # background
    pos_mask = max_iou >= pos_iou
    labels[pos_mask] = gt_labels[max_idx[pos_mask]]

    # Force-match: each GT box's argmax-IoU anchor is positive (helps tiny GT recall).
    forced = iou.argmax(dim=0)  # (M,) - best anchor per GT
    labels[forced] = gt_labels  # override regardless of IoU
    max_idx[forced] = torch.arange(gt_boxes.shape[0], device=anchors.device)

    matched = torch.full((n,), -1, dtype=torch.long, device=anchors.device)
    is_pos = labels >= 0
    matched[is_pos] = max_idx[is_pos]
    return labels, matched
