"""Loss helpers used by the multi-task model.

We keep them here so ``train.py`` stays compact:
- ``sigmoid_focal_loss``: classification loss for the dense detection head
  (Lin et al., 2017, Focal Loss for Dense Object Detection).
- ``dice_loss``: complement to BCE for segmentation; helps when foreground is sparse.
- ``smooth_l1_loss``: standard regression loss for box deltas.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Focal loss with sigmoid activation. Targets are 0/1 of same shape as logits."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - pt).pow(gamma) * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation. logits/targets shape: (N, 1, H, W)."""
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def smooth_l1(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0 / 9.0) -> torch.Tensor:
    """Element-wise smooth L1, returns tensor the same shape; sum/mean is up to caller."""
    diff = pred - target
    abs_diff = diff.abs()
    return torch.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
