# Design Note

## Architecture

```
                                                  ┌────────────────┐
                                                  │ Segmentation   │
                                          ┌──────►│   head         │──► (B,1,H,W)
                                          │       │ (Panoptic-FPN) │
                                          │       └────────────────┘
       ┌─────────────┐       ┌──────────┐ │
input ─►│ ResNet-50   │──C2─► │          │ │       ┌────────────────┐
        │ (ImageNet)  │──C3─► │   FPN    │─┼──────►│ Detection head │──► boxes
        │             │──C4─► │ (P2..P5, │ │       │ (RetinaNet,    │   labels
        │             │──C5─► │  P6, P7) │ │       │  P3..P7,       │   scores
        └──────┬──────┘       └──────────┘ │       │  9 anchors/loc)│
               │                           │       └────────────────┘
               │                           │
               │     C5  ─────────────►    │       ┌────────────────┐
               └─────────────────────────────────► │ Classification │──► state
                                                   │  head (GAP+FC) │   (5 classes)
                                                   └────────────────┘
```

## How the model works:

The model can do **three tasks**:
- Segmentation (finding regions in the image)
- Detection (finding objects)
- Classification (labeling the whole image)

But it doesn’t run everything at once.  
It only runs the part needed for the current task, which makes it faster.

All tasks share the same **backbone (feature extractor)**, so learning from one task helps improve the others.

---

## 🤔 Why FPN instead of BiFPN?

Both help the model understand features at different scales.

I chose **FPN** because:
- The dataset is **very small**, so BiFPN may overfit
- FPN is **simpler and easier to debug**
- It’s **well-tested** (used in RetinaNet)

We can switch to BiFPN later if needed.

---

## 🔧 How each task is handled

### 1. Segmentation
- Combines features from multiple layers
- Upsamples them to match the image size
- Predicts pixel-wise output (which part belongs to object)

### 2. Detection
- Uses anchor boxes to find objects
- Predicts:
  - Object class
  - Bounding box location
- Uses **focal loss** to handle class imbalance

### 3. Classification
- Uses deep features of the image
- Outputs a single label for the whole image

---

## ⚖️ How training is balanced

### 1. Task sampling
Each training step randomly picks one task.  
This ensures all tasks get equal importance.

### 2. Loss balancing
All task losses are similar in scale, so we simply add them.  
(No complex weighting needed)

---

## 🏋️ How training is done

- One model, one optimizer
- Each step trains only one task
- Backbone is shared across all tasks
- BatchNorm layers are frozen (helps with small data)
- Each task has its own dataset

---

## 🚀 What can be improved

- Add **data augmentation** (flip, color jitter, etc.)
- Use proper detection metrics like **mAP**
- Use better loss balancing (e.g., uncertainty-based)
- Combine segmentation and detection data
- Handle class imbalance better
- Try **BiFPN** for better accuracy
- Use **mixed precision** for faster training

---

## 💡 Summary

This is a **multi-task model** that:
- Shares learning across tasks
- Trains efficiently
- Keeps the design simple and practical