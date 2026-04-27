"""Run all three task heads on a folder of images and save visualizations.

Usage::

    python inference.py --checkpoint checkpoints/last.pt \
        --inputs data/assignment-dataset/segmentation/images \
        --output-dir sample_outputs --num-images 5

Each input image gets a single composite figure with three panels:
    1. segmentation overlay (foreground in red)
    2. detection boxes (per-class color, score in label)
    3. predicted cooking-state probabilities as a horizontal bar
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.patches import Rectangle

from model.multitask import MultiTaskModel
from mtl_data.cls_dataset import INDEX_TO_LABEL
from mtl_data.det_dataset import DET_CLASS_NAMES
from mtl_data.transforms import denormalize, letterbox, normalize, to_tensor


def load_model(checkpoint: str, device: str) -> MultiTaskModel:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MultiTaskModel(
        num_seg_classes=1,
        num_cls_classes=5,
        num_det_classes=3,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model


def collect_images(inputs: List[str], num_images: int) -> List[Path]:
    paths: List[Path] = []
    for entry in inputs:
        p = Path(entry)
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                paths.extend(sorted(p.glob(ext)))
        elif p.is_file():
            paths.append(p)
    if num_images > 0:
        paths = paths[:num_images]
    return paths


def visualize_one(
    img_pil: Image.Image,
    seg_prob: np.ndarray,
    det: dict,
    cls_probs: np.ndarray,
    output_path: Path,
    title: str = "",
) -> None:
    """Compose a 3-panel figure and save."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    img = np.array(img_pil)

    # Panel 1: segmentation overlay
    ax = axes[0]
    ax.imshow(img)
    overlay = np.zeros_like(img)
    overlay[..., 0] = (seg_prob * 255).astype(np.uint8)
    ax.imshow(overlay, alpha=0.4)
    ax.set_title("Segmentation")
    ax.axis("off")

    # Panel 2: detection
    ax = axes[1]
    ax.imshow(img)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                         edgecolor=colors[label % len(colors)], linewidth=2)
        ax.add_patch(rect)
        name = DET_CLASS_NAMES[label] if label < len(DET_CLASS_NAMES) else f"cls{label}"
        ax.text(x1, max(y1 - 4, 0), f"{name} {score:.2f}",
                color="white", fontsize=8,
                bbox=dict(facecolor=colors[label % len(colors)], alpha=0.7, pad=1))
    ax.set_title(f"Detection ({len(det['boxes'])} boxes)")
    ax.axis("off")

    # Panel 3: cooking-state bar
    ax = axes[2]
    labels = [INDEX_TO_LABEL[i] for i in range(len(cls_probs))]
    bars = ax.barh(labels, cls_probs)
    pred_idx = int(cls_probs.argmax())
    bars[pred_idx].set_color("#d62728")
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title(f"Cooking state -> {labels[pred_idx]}")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run(args):
    device = args.device
    model = load_model(args.checkpoint, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_images(args.inputs, args.num_images)
    if not paths:
        print(f"No images found under: {args.inputs}")
        return

    for path in paths:
        orig = Image.open(path).convert("RGB")
        img_lb, _, _, info = letterbox(orig, args.image_size)
        img_t = normalize(to_tensor(img_lb)).unsqueeze(0).to(device)

        out = model(img_t, task="all")
        # Segmentation prob (binary)
        seg_prob = torch.sigmoid(out.seg_logits)[0, 0].cpu().numpy()
        # Cls
        cls_probs = torch.softmax(out.cls_logits, dim=1)[0].cpu().numpy()
        # Det (use built-in NMS pipeline)
        det_results = model.predict_detections(
            img_t, score_thresh=args.score_thresh, nms_iou=args.nms_iou,
        )[0]
        det = {
            "boxes": det_results["boxes"].cpu().numpy(),
            "scores": det_results["scores"].cpu().numpy(),
            "labels": det_results["labels"].cpu().numpy(),
        }

        visualize_one(
            img_pil=img_lb,
            seg_prob=seg_prob,
            det=det,
            cls_probs=cls_probs,
            output_path=out_dir / f"{path.stem}.png",
            title=f"{path.name}",
        )
        print(f"[viz] {path.name} -> {out_dir / (path.stem + '.png')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/last.pt")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more image files or directories.")
    p.add_argument("--output-dir", default="sample_outputs")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--num-images", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
