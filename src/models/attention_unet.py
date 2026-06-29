"""Attention U-Net (proposal section 3.1.2, Oktay et al. 2018).

The Attention U-Net adds *attention gates* to the skip connections of a standard
U-Net. Before the encoder features are concatenated into the decoder, each gate
uses the coarser decoder signal to compute a soft attention map and re-weights
the skip features — suppressing irrelevant background and emphasising the tumour
region.

It is built on the SAME from-scratch backbone as ``BaselineUNet`` so that the
only difference is the attention gating. Comparing the two therefore isolates the
contribution of attention, exactly the controlled experiment the proposal asks
for ("assess the impact of attention gating on segmentation precision").

Reference:
    Oktay, O. et al. (2018) 'Attention U-Net: Learning where to look for the
    pancreas', arXiv:1804.03999.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import _DoubleConv


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay 2018).

    ``g`` is the gating signal (decoder feature at the skip resolution) and ``x``
    is the encoder skip feature. Returns ``x`` multiplied by the learned
    attention coefficients (same shape as ``x``).
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        attn = self.relu(self.w_g(g) + self.w_x(x))
        attn = self.psi(attn)                 # (B, 1, H, W) coefficients in [0, 1]
        return x * attn


class AttentionUNet(nn.Module):
    """4-level U-Net with an attention gate on every skip connection."""

    def __init__(self, in_channels: int = 3, classes: int = 1,
                 features=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # encoder
        self.downs = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(_DoubleConv(ch, f))
            ch = f

        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        # decoder: per level -> upconv, attention gate, double conv
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.gates.append(AttentionGate(f_g=f, f_l=f, f_int=f // 2))
            self.ups.append(_DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)                       # transposed conv (gating signal)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:])
            skip = self.gates[i // 2](g=x, x=skip)   # attention-gated skip
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)                   # double conv

        return self.head(x)
