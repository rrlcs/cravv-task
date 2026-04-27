"""Multi-task training entry point.

Strategy
--------
- Each task has its own DataLoader (different image counts and label types).
- We iterate ``--steps-per-epoch`` steps per epoch. At each step we draw a task
  according to a fixed sampling distribution (default: each task = 1/3) and
  pull one batch from that task's loader. This avoids variable-length-batch
  hassles of mixing tasks within a single batch and keeps gradient signal
  balanced across tasks per epoch.
- The shared backbone & FPN see gradients from all tasks. Each task has its
  own loss; we apply per-task scalar weights (``--seg-w / --cls-w / --det-w``)
  before backward.
- Validation reports per-task metrics (Dice, top-1, mAP-lite proxy via mean
  classification accuracy of positive anchors) using held-out splits.

This script is intentionally minimal - it is not a research-grade trainer.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model.anchors import encode_boxes, match_anchors_to_targets
from model.losses import dice_loss, sigmoid_focal_loss, smooth_l1
from model.multitask import MultiTaskModel
from mtl_data import (
    ClassificationDataset,
    DetectionDataset,
    SegmentationDataset,
    det_collate_fn,
)


# ---------------------------------------------------------------------------
# Per-task losses
# ---------------------------------------------------------------------------

def seg_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE + Dice for binary segmentation. Both (B, 1, H, W)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice


def cls_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def det_loss(
    cls_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    anchors_per_level: List[torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Compute focal + smooth L1 loss for the dense detection head.

    Returns (cls_loss, box_loss, num_positive_anchors)
    """
    anchors = torch.cat(anchors_per_level, dim=0)  # (A, 4)
    device = cls_logits.device
    B, A, C = cls_logits.shape
    assert A == anchors.shape[0]

    cls_targets = torch.zeros((B, A, C), device=device)
    box_targets = torch.zeros((B, A, 4), device=device)
    pos_mask = torch.zeros((B, A), dtype=torch.bool, device=device)
    valid_mask = torch.ones((B, A), dtype=torch.bool, device=device)  # excludes "ignore"

    for i in range(B):
        gt_boxes = targets[i]["boxes"].to(device)
        gt_labels = targets[i]["labels"].to(device)
        labels, matched = match_anchors_to_targets(anchors, gt_boxes, gt_labels)

        # Ignored anchors get 0 in cls target and don't contribute - exclude via valid_mask.
        ignored = labels == -2
        valid_mask[i, ignored] = False

        positives = labels >= 0
        if positives.any():
            pos_mask[i, positives] = True
            cls_targets[i, positives, labels[positives]] = 1.0
            matched_boxes = gt_boxes[matched[positives]]
            box_targets[i, positives] = encode_boxes(anchors[positives], matched_boxes)

    valid_cls = valid_mask.unsqueeze(-1).expand_as(cls_targets)
    cls_l = sigmoid_focal_loss(
        cls_logits[valid_cls],
        cls_targets[valid_cls],
        reduction="sum",
    )

    num_pos = max(int(pos_mask.sum().item()), 1)
    cls_l = cls_l / num_pos

    if pos_mask.any():
        box_l = smooth_l1(box_deltas[pos_mask], box_targets[pos_mask]).sum() / num_pos
    else:
        box_l = box_deltas.sum() * 0.0  # zero with grad attached

    return cls_l, box_l, num_pos


# ---------------------------------------------------------------------------
# Splits + loaders
# ---------------------------------------------------------------------------

def make_split(n: int, val_frac: float, seed: int = 0):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:], idx[:n_val]


def build_loaders(args):
    seg_full = SegmentationDataset(args.data_root, image_size=args.image_size)
    cls_full = ClassificationDataset(args.data_root, image_size=args.image_size)
    det_full = DetectionDataset(args.data_root, image_size=args.image_size)

    if args.subset_frac < 1.0:
        # Sub-sample each task's available data deterministically.
        rng = random.Random(args.seed)
        def _sub(ds):
            n = len(ds)
            keep = max(args.min_per_task, int(n * args.subset_frac))
            keep = min(keep, n)
            idx = list(range(n))
            rng.shuffle(idx)
            return Subset(ds, idx[:keep])
        seg_full = _sub(seg_full)
        cls_full = _sub(cls_full)
        det_full = _sub(det_full)

    seg_train_idx, seg_val_idx = make_split(len(seg_full), args.val_frac, args.seed)
    cls_train_idx, cls_val_idx = make_split(len(cls_full), args.val_frac, args.seed + 1)
    det_train_idx, det_val_idx = make_split(len(det_full), args.val_frac, args.seed + 2)

    seg_tr, seg_va = Subset(seg_full, seg_train_idx), Subset(seg_full, seg_val_idx)
    cls_tr, cls_va = Subset(cls_full, cls_train_idx), Subset(cls_full, cls_val_idx)
    det_tr, det_va = Subset(det_full, det_train_idx), Subset(det_full, det_val_idx)

    print(f"[data] seg: train={len(seg_tr)} val={len(seg_va)}")
    print(f"[data] cls: train={len(cls_tr)} val={len(cls_va)}")
    print(f"[data] det: train={len(det_tr)} val={len(det_va)}")

    loaders = {
        "seg_train": DataLoader(seg_tr, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True),
        "cls_train": DataLoader(cls_tr, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True),
        "det_train": DataLoader(det_tr, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True,
                                collate_fn=det_collate_fn),
        "seg_val": DataLoader(seg_va, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers),
        "cls_val": DataLoader(cls_va, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers),
        "det_val": DataLoader(det_va, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=det_collate_fn),
    }
    return loaders


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: MultiTaskModel, loaders, device: str) -> Dict[str, float]:
    model.eval()
    metrics = {}

    # Segmentation: mean Dice over val set.
    dices = []
    for img, mask in loaders["seg_val"]:
        img = img.to(device); mask = mask.to(device)
        out = model(img, task="seg")
        prob = torch.sigmoid(out.seg_logits) > 0.5
        prob = prob.float().flatten(1)
        m = mask.flatten(1)
        inter = (prob * m).sum(1)
        denom = prob.sum(1) + m.sum(1)
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        dices.append(dice.mean().item())
    metrics["seg/dice"] = sum(dices) / max(len(dices), 1)

    # Classification: top-1 accuracy.
    correct = total = 0
    for img, label in loaders["cls_val"]:
        img = img.to(device); label = label.to(device)
        out = model(img, task="cls")
        pred = out.cls_logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()
    metrics["cls/acc"] = correct / max(total, 1)

    # Detection: how often the model puts at least one box on each annotated image
    # (we use a coarse "image-level recall" proxy at score>=0.3, NMS=0.5).
    hits = total = 0
    for img, targets in loaders["det_val"]:
        img = img.to(device)
        preds = model.predict_detections(img, score_thresh=0.3, nms_iou=0.5)
        for p, t in zip(preds, targets):
            if t["boxes"].numel() > 0:
                total += 1
                if p["boxes"].numel() > 0:
                    hits += 1
    metrics["det/has_box_when_gt"] = hits / max(total, 1)

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = args.device
    print(f"[init] device={device}")

    loaders = build_loaders(args)
    model = MultiTaskModel(
        num_seg_classes=1,
        num_cls_classes=5,
        num_det_classes=3,
        pretrained=not args.no_pretrained,
    ).to(device)

    # Train all params except frozen BN (already frozen inside the backbone).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch
    )

    # Endless cycles for random task sampling.
    train_iters = {
        "seg": cycle(iter(loaders["seg_train"])),
        "cls": cycle(iter(loaders["cls_train"])),
        "det": cycle(iter(loaders["det_train"])),
    }
    task_names = ["seg", "cls", "det"]
    task_weights = {"seg": args.seg_w, "cls": args.cls_w, "det": args.det_w}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    log_every = max(args.log_every, 1)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running = {"seg": 0.0, "cls": 0.0, "det_cls": 0.0, "det_box": 0.0}
        counts = {"seg": 0, "cls": 0, "det": 0}
        t0 = time.time()
        for step in range(args.steps_per_epoch):
            task = rng.choice(task_names)
            optimizer.zero_grad()

            if task == "seg":
                img, mask = next(train_iters["seg"])
                img = img.to(device); mask = mask.to(device)
                out = model(img, task="seg")
                loss_main = seg_loss(out.seg_logits, mask)
                loss_total = task_weights["seg"] * loss_main
                running["seg"] += loss_main.item()
                counts["seg"] += 1
            elif task == "cls":
                img, label = next(train_iters["cls"])
                img = img.to(device); label = label.to(device)
                out = model(img, task="cls")
                loss_main = cls_loss(out.cls_logits, label)
                loss_total = task_weights["cls"] * loss_main
                running["cls"] += loss_main.item()
                counts["cls"] += 1
            else:  # det
                img, targets = next(train_iters["det"])
                img = img.to(device)
                out = model(img, task="det")
                cls_l, box_l, _ = det_loss(
                    out.det_cls_logits, out.det_box_deltas, out.det_anchors,
                    targets, num_classes=model.num_det_classes,
                )
                loss_main = cls_l + box_l
                loss_total = task_weights["det"] * loss_main
                running["det_cls"] += cls_l.item()
                running["det_box"] += box_l.item()
                counts["det"] += 1

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if (step + 1) % log_every == 0:
                ce_seg = running["seg"] / max(counts["seg"], 1)
                ce_cls = running["cls"] / max(counts["cls"], 1)
                ce_dc = running["det_cls"] / max(counts["det"], 1)
                ce_db = running["det_box"] / max(counts["det"], 1)
                print(
                    f"[train] epoch={epoch} step={step+1}/{args.steps_per_epoch} "
                    f"task={task} | seg={ce_seg:.3f} cls={ce_cls:.3f} "
                    f"det_cls={ce_dc:.3f} det_box={ce_db:.3f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        dt = time.time() - t0
        print(f"[train] epoch {epoch} done in {dt:.1f}s")

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            metrics = validate(model, loaders, device)
            print("[val] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        # Always save the last checkpoint; saving every epoch is cheap and means
        # smoke tests or interrupted runs still produce a usable checkpoint.
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")
        print(f"[ckpt] saved -> {out_dir / 'last.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/assignment-dataset")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=60)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Task-loss weights. Defaults pick comparable magnitudes empirically:
    # det loss tends to be smaller per step early-on, so we boost it slightly.
    p.add_argument("--seg-w", type=float, default=1.0)
    p.add_argument("--cls-w", type=float, default=1.0)
    p.add_argument("--det-w", type=float, default=1.0)
    p.add_argument("--subset-frac", type=float, default=1.0,
                   help="Use only this fraction of each task's dataset (for smoke runs).")
    p.add_argument("--min-per-task", type=int, default=8,
                   help="Minimum number of items per task when subsetting.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip ImageNet pretrained weights (for offline smoke tests).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)
