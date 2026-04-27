"""ResNet-50 backbone exposing C2-C5 feature maps.

We rely on torchvision's ImageNet-pretrained ResNet-50 and tap features
right after each "layer" block. Strides relative to input:
    C2: 1/4    (after layer1, 256 channels)
    C3: 1/8    (after layer2, 512 channels)
    C4: 1/16   (after layer3, 1024 channels)
    C5: 1/32   (after layer4, 2048 channels)
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    """Wrap torchvision's ResNet-50 and return a dict of C2..C5 feature maps."""

    out_channels: Dict[str, int] = {
        "C2": 256,
        "C3": 512,
        "C4": 1024,
        "C5": 2048,
    }

    def __init__(self, pretrained: bool = True, freeze_bn: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)

        # Stem: conv1 -> bn1 -> relu -> maxpool reduces spatial dim by 4 already.
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
        )
        self.layer1 = net.layer1  # -> C2 (1/4, 256ch)
        self.layer2 = net.layer2  # -> C3 (1/8, 512ch)
        self.layer3 = net.layer3  # -> C4 (1/16, 1024ch)
        self.layer4 = net.layer4  # -> C5 (1/32, 2048ch)

        if freeze_bn:
            self._freeze_batchnorm()

    def _freeze_batchnorm(self) -> None:
        """Freeze BN running stats. Common practice when fine-tuning detectors
        with small batch sizes; BN running stats from ImageNet are kept."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep BN frozen even after .train() flips the flag.
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict(C2=c2, C3=c3, C4=c4, C5=c5)
