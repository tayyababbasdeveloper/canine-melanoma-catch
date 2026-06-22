"""U-Net segmentation models (proposal section 3.1.2).

Two architectures are provided so the proposal's plan — "beginning with a
baseline architecture before incorporating the ResNet-34 encoder" — can be
followed exactly:

  * ``BaselineUNet`` — a from-scratch U-Net (Ronneberger et al., 2015) with no
    pretraining, as a controlled baseline.
  * ``build_smp_unet`` — segmentation-models-pytorch U-Net with an
    ImageNet-pretrained ResNet-34 encoder (transfer learning).

``build_model(cfg)`` selects between them from the config and degrades
gracefully to randomly-initialised weights if pretrained weights cannot be
downloaded (offline environments).
"""
from __future__ import annotations

import torch
import torch.nn as nn


# =====================================================================
# Baseline U-Net (from scratch)
# =====================================================================
class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BaselineUNet(nn.Module):
    """Classic 4-level U-Net with skip connections."""

    def __init__(self, in_channels: int = 3, classes: int = 1,
                 features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        ch = in_channels
        for f in features:
            self.downs.append(_DoubleConv(ch, f))
            ch = f

        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
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
            x = self.ups[i](x)               # transposed conv
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)           # double conv

        return self.head(x)


# =====================================================================
# Transfer-learning U-Net (segmentation-models-pytorch)
# =====================================================================
def build_smp_unet(encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
                   in_channels: int = 3, classes: int = 1, logger=None) -> nn.Module:
    """U-Net with a pretrained encoder; falls back to random weights if offline."""
    import segmentation_models_pytorch as smp

    try:
        model = smp.Unet(
            encoder_name=encoder, encoder_weights=encoder_weights,
            in_channels=in_channels, classes=classes,
        )
    except Exception as exc:  # network/download failure -> train from scratch
        msg = (f"Could not load '{encoder_weights}' weights for {encoder} "
               f"({exc}); falling back to randomly-initialised encoder.")
        if logger:
            logger.warning(msg)
        else:
            print("[WARN]", msg)
        model = smp.Unet(
            encoder_name=encoder, encoder_weights=None,
            in_channels=in_channels, classes=classes,
        )
    return model


def build_model(cfg: dict, logger=None) -> nn.Module:
    """Factory: build the segmentation model described in the config."""
    s = cfg["segmentation"]
    arch = s.get("arch", "unet")
    if arch == "baseline":
        return BaselineUNet(in_channels=s.get("in_channels", 3),
                            classes=s.get("classes", 1))
    return build_smp_unet(
        encoder=s.get("encoder", "resnet34"),
        encoder_weights=s.get("encoder_weights", "imagenet"),
        in_channels=s.get("in_channels", 3),
        classes=s.get("classes", 1),
        logger=logger,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
