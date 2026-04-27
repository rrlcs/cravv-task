"""Feature Pyramid Network on top of ResNet C2-C5.

Produces P2..P5 (top-down + lateral) and optionally P6/P7 for detection.
Following Lin et al., 2017:
    Pk = lateral_k(Ck) + Upsample(Pk+1)
    output convs are 3x3 to clean up aliasing from upsampling.
P6 = stride-2 conv on C5
P7 = stride-2 conv on P6 with ReLU in between (RetinaNet style)
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(
        self,
        in_channels_per_level: Dict[str, int],
        out_channels: int = 256,
        extra_blocks: bool = True,
    ) -> None:
        """
        Args:
            in_channels_per_level: e.g. {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}
            out_channels: feature channels of every pyramid level (commonly 256)
            extra_blocks: if True, also produce P6/P7 used by RetinaNet-style detectors.
        """
        super().__init__()
        self.out_channels = out_channels
        self.in_levels: List[str] = list(in_channels_per_level.keys())  # ["C2", "C3", "C4", "C5"]
        self.out_levels: List[str] = ["P2", "P3", "P4", "P5"]
        self.extra_blocks = extra_blocks

        self.lateral_convs = nn.ModuleDict()
        self.output_convs = nn.ModuleDict()
        for c_name, p_name in zip(self.in_levels, self.out_levels):
            in_ch = in_channels_per_level[c_name]
            self.lateral_convs[p_name] = nn.Conv2d(in_ch, out_channels, kernel_size=1)
            self.output_convs[p_name] = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if extra_blocks:
            # P6 from C5 (stride 2 conv 3x3), P7 from P6 (relu + stride 2 conv 3x3)
            self.p6 = nn.Conv2d(in_channels_per_level["C5"], out_channels, kernel_size=3, stride=2, padding=1)
            self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.out_levels = self.out_levels + ["P6", "P7"]

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c_feats: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
        # Top-down: start at C5
        # Build laterals first
        laterals = {p: self.lateral_convs[p](c_feats[c]) for c, p in zip(self.in_levels, ["P2", "P3", "P4", "P5"])}

        # Top-down accumulate
        # P5 has no upsampled contribution.
        # P4 = lateral(C4) + upsample(P5_top), etc.
        merged = {"P5": laterals["P5"]}
        for higher, lower in [("P5", "P4"), ("P4", "P3"), ("P3", "P2")]:
            up = F.interpolate(merged[higher], size=laterals[lower].shape[-2:], mode="nearest")
            merged[lower] = laterals[lower] + up

        outs = OrderedDict()
        for p in ["P2", "P3", "P4", "P5"]:
            outs[p] = self.output_convs[p](merged[p])

        if self.extra_blocks:
            p6 = self.p6(c_feats["C5"])
            p7 = self.p7(F.relu(p6))
            outs["P6"] = p6
            outs["P7"] = p7

        return outs
