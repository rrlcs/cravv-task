"""The multi-task model: ResNet-50 + FPN + (segmentation, classification, detection) heads.

The key design choice is task-selective forward: ``forward(images, task=...)``
only runs the head(s) needed for that step. This keeps memory low when one
batch's task is segmentation only, and it makes the round-robin training loop
trivial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .anchors import AnchorGenerator, decode_boxes
from .backbone import ResNet50Backbone
from .fpn import FPN
from .heads import ClassificationHead, DetectionHead, SegmentationHead


# Order of FPN levels used by detection (RetinaNet uses P3-P7).
DET_LEVELS: Tuple[str, ...] = ("P3", "P4", "P5", "P6", "P7")
DET_STRIDES: Tuple[int, ...] = (8, 16, 32, 64, 128)
DET_SIZES: Tuple[float, ...] = (32.0, 64.0, 128.0, 256.0, 512.0)


@dataclass
class MultiTaskOutput:
    """Container for whichever head outputs were requested in a forward."""
    seg_logits: Optional[torch.Tensor] = None  # (B, num_seg_classes, H, W)
    cls_logits: Optional[torch.Tensor] = None  # (B, num_cls_classes)
    det_cls_logits: Optional[torch.Tensor] = None  # (B, A, num_det_classes)
    det_box_deltas: Optional[torch.Tensor] = None  # (B, A, 4)
    det_anchors: Optional[List[torch.Tensor]] = None  # list per level
    det_levels: Optional[List[str]] = None
    det_feature_sizes: Optional[List[Tuple[int, int]]] = None


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        num_seg_classes: int = 1,
        num_cls_classes: int = 5,
        num_det_classes: int = 3,
        fpn_out_channels: int = 256,
        seg_inner_channels: int = 128,
        anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        anchor_scales: Tuple[float, ...] = (1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        self.num_det_classes = num_det_classes

        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.fpn = FPN(self.backbone.out_channels, out_channels=fpn_out_channels, extra_blocks=True)

        self.seg_head = SegmentationHead(
            in_channels=fpn_out_channels,
            inner_channels=seg_inner_channels,
            num_classes=num_seg_classes,
            levels=("P2", "P3", "P4", "P5"),
        )
        self.cls_head = ClassificationHead(
            in_channels=self.backbone.out_channels["C5"],
            num_classes=num_cls_classes,
        )
        num_anchors = len(anchor_ratios) * len(anchor_scales)
        self.det_head = DetectionHead(
            in_channels=fpn_out_channels,
            num_classes=num_det_classes,
            num_anchors_per_loc=num_anchors,
        )
        self.anchor_generator = AnchorGenerator(
            levels=list(DET_LEVELS),
            strides=list(DET_STRIDES),
            sizes=list(DET_SIZES),
            ratios=anchor_ratios,
            scales=anchor_scales,
        )

    # ------------------------------------------------------------------ utils
    @property
    def det_levels(self) -> List[str]:
        return list(DET_LEVELS)

    def get_anchors(self, feature_sizes: List[Tuple[int, int]], device: torch.device) -> List[torch.Tensor]:
        return self.anchor_generator(feature_sizes, device)

    # ------------------------------------------------------------- forward
    def forward(self, images: torch.Tensor, task: str = "all") -> MultiTaskOutput:
        """Run a forward pass for one or all tasks.

        Args:
            images: (B, 3, H, W) tensor, ImageNet-normalized.
            task: one of {"seg", "cls", "det", "all"}.
        """
        assert task in {"seg", "cls", "det", "all"}, f"Unknown task: {task}"

        c_feats = self.backbone(images)
        out = MultiTaskOutput()

        # FPN is needed for seg + det. We only run it when needed.
        need_fpn = task in {"seg", "det", "all"}
        p_feats = self.fpn(c_feats) if need_fpn else None

        if task in {"seg", "all"}:
            out.seg_logits = self.seg_head(p_feats, out_size=images.shape[-2:])

        if task in {"cls", "all"}:
            out.cls_logits = self.cls_head(c_feats["C5"])

        if task in {"det", "all"}:
            cls_logits, box_deltas, sizes = self.det_head(p_feats, list(DET_LEVELS))
            anchors = self.get_anchors(sizes, images.device)
            out.det_cls_logits = cls_logits
            out.det_box_deltas = box_deltas
            out.det_anchors = anchors
            out.det_levels = list(DET_LEVELS)
            out.det_feature_sizes = sizes

        return out

    # ------------------------------------------------------------ inference helpers
    @torch.no_grad()
    def predict_detections(
        self,
        images: torch.Tensor,
        score_thresh: float = 0.3,
        nms_iou: float = 0.5,
        max_per_image: int = 100,
        pre_nms_top_k: int = 1000,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run detection inference and return per-image decoded boxes."""
        from torchvision.ops import batched_nms

        out = self.forward(images, task="det")
        anchors_per_level = out.det_anchors
        anchors = torch.cat(anchors_per_level, dim=0)  # (A, 4)

        H, W = images.shape[-2:]
        results = []
        B = images.shape[0]
        for i in range(B):
            cls_logit = out.det_cls_logits[i]  # (A, C)
            box_delta = out.det_box_deltas[i]  # (A, 4)
            scores = cls_logit.sigmoid()

            # Flatten across (A, C) to get a candidate per (anchor, class).
            num_anchors, num_classes = scores.shape
            flat_scores = scores.reshape(-1)
            keep = flat_scores > score_thresh
            if keep.sum() == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=images.device),
                    "scores": torch.zeros((0,), device=images.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=images.device),
                })
                continue

            cand_idx = keep.nonzero(as_tuple=False).squeeze(1)
            # Top-k pre-NMS to keep cost bounded.
            if cand_idx.numel() > pre_nms_top_k:
                top_scores, top_pos = flat_scores[cand_idx].topk(pre_nms_top_k)
                cand_idx = cand_idx[top_pos]

            anchor_idx = cand_idx // num_classes
            class_idx = cand_idx % num_classes
            cand_scores = flat_scores[cand_idx]

            cand_anchors = anchors[anchor_idx]
            cand_deltas = box_delta[anchor_idx]
            cand_boxes = decode_boxes(cand_anchors, cand_deltas)
            cand_boxes[:, [0, 2]] = cand_boxes[:, [0, 2]].clamp(0, W - 1)
            cand_boxes[:, [1, 3]] = cand_boxes[:, [1, 3]].clamp(0, H - 1)

            keep_idx = batched_nms(cand_boxes, cand_scores, class_idx, nms_iou)
            keep_idx = keep_idx[:max_per_image]

            results.append({
                "boxes": cand_boxes[keep_idx],
                "scores": cand_scores[keep_idx],
                "labels": class_idx[keep_idx],
            })
        return results
