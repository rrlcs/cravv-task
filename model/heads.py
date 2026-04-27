"""The three task heads.

All heads are designed to consume the FPN features produced by ``model.fpn.FPN``
(plus C5 directly for the classification head, since FPN at P5 is fine too but
C5 keeps the cls head pretrained-weight friendly).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Segmentation head (Panoptic-FPN style)
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Panoptic-FPN style semantic segmentation head.

    Each FPN level passes through a small (3x3 conv + GN + ReLU) stack and is
    bilinearly upsampled to ``out_stride`` resolution. The four streams are
    summed and projected to ``num_classes`` logits, then upsampled to the
    network input size by the caller (or here, optionally).
    """

    def __init__(
        self,
        in_channels: int = 256,
        inner_channels: int = 128,
        num_classes: int = 1,
        levels: Tuple[str, ...] = ("P2", "P3", "P4", "P5"),
        level_strides: Dict[str, int] = None,
        out_stride: int = 4,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.out_stride = out_stride
        if level_strides is None:
            level_strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}
        self.level_strides = level_strides

        self.branches = nn.ModuleDict()
        for lvl in levels:
            stride = level_strides[lvl]
            num_upsamples = max(int(math.log2(stride // out_stride)), 0)
            layers = []
            ch_in = in_channels
            for _ in range(max(num_upsamples, 1)):
                layers += [
                    nn.Conv2d(ch_in, inner_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, inner_channels),
                    nn.ReLU(inplace=True),
                ]
                ch_in = inner_channels
            self.branches[lvl] = nn.Sequential(*layers)

        self.predictor = nn.Conv2d(inner_channels, num_classes, kernel_size=1)

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        out_size: Tuple[int, int] = None,
    ) -> torch.Tensor:
        """Returns logits at full input resolution if ``out_size`` is given,
        else at ``out_stride`` (defaults to 1/4)."""
        target_lvl = self.levels[0]  # P2 -> 1/out_stride
        target_hw = feats[target_lvl].shape[-2:]

        merged = None
        for lvl in self.levels:
            x = self.branches[lvl](feats[lvl])
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            merged = x if merged is None else merged + x

        logits = self.predictor(merged)
        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# State (cooking) classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Global pooled MLP on the deepest backbone feature map (C5).

    Using C5 directly (not FPN P5) keeps a clean separation: the classification
    label is image-level, so we don't need spatial pyramid features.
    """

    def __init__(self, in_channels: int = 2048, num_classes: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, c5: torch.Tensor) -> torch.Tensor:
        x = self.pool(c5).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Detection head (RetinaNet-style)
# ---------------------------------------------------------------------------

class RetinaSubnet(nn.Module):
    """4 x (3x3 conv + ReLU) followed by a final prediction conv. Shared across levels."""

    def __init__(self, in_channels: int, mid_channels: int, num_outputs: int, num_layers: int = 4,
                 prior_prob: float = None) -> None:
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers += [
                nn.Conv2d(ch, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            ch = mid_channels
        self.tower = nn.Sequential(*layers)
        self.predictor = nn.Conv2d(mid_channels, num_outputs, kernel_size=3, padding=1)

        for m in self.tower.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.predictor.weight, std=0.01)
        if prior_prob is not None:
            # Per RetinaNet, init final cls bias so initial sigmoid output ~ prior_prob.
            bias = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.predictor.bias, bias)
        else:
            nn.init.zeros_(self.predictor.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.tower(x))


class DetectionHead(nn.Module):
    """Dense single-stage detector head: shared cls + reg subnets across FPN levels."""

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        num_anchors_per_loc: int = 9,
        mid_channels: int = 256,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_loc
        self.cls_subnet = RetinaSubnet(
            in_channels, mid_channels,
            num_outputs=num_anchors_per_loc * num_classes,
            prior_prob=prior_prob,
        )
        self.reg_subnet = RetinaSubnet(
            in_channels, mid_channels,
            num_outputs=num_anchors_per_loc * 4,
        )

    def forward(self, feats: Dict[str, torch.Tensor], levels: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """Returns:
            cls_logits: (B, A_total, num_classes)
            box_deltas: (B, A_total, 4)
            feature_sizes: list of (H, W) per level (in same order as ``levels``)
        """
        cls_outs, reg_outs, sizes = [], [], []
        for lvl in levels:
            x = feats[lvl]
            B, _, H, W = x.shape
            cls = self.cls_subnet(x)  # (B, K*C, H, W)
            reg = self.reg_subnet(x)  # (B, K*4, H, W)

            # Reshape so that anchors are the second-to-last dim.
            cls = cls.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, self.num_classes)
            reg = reg.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, 4)
            cls_outs.append(cls)
            reg_outs.append(reg)
            sizes.append((H, W))

        cls_logits = torch.cat(cls_outs, dim=1)
        box_deltas = torch.cat(reg_outs, dim=1)
        return cls_logits, box_deltas, sizes
