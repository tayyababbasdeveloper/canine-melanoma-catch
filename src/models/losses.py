"""Loss functions for tumour segmentation.

Proposal section 3.1.2 specifies a composite loss of binary cross-entropy and
Dice loss. BCE gives stable per-pixel gradients; Dice directly optimises region
overlap and is robust to the foreground/background imbalance typical of tumour
masks. The combination is the de-facto standard for medical segmentation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss on sigmoid probabilities (1 - Dice coefficient)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice loss.

    ``pos_weight`` up-weights the (rare) tumour pixels in the BCE term to counter
    class imbalance.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def build_loss(cfg: dict, pos_weight: float | None = None) -> nn.Module:
    """Construct the segmentation loss from the config."""
    s = cfg["segmentation"]
    name = s.get("loss", "bce_dice")
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=s.get("bce_weight", 0.5),
            dice_weight=s.get("dice_weight", 0.5),
            pos_weight=pos_weight,
        )
    if name == "dice":
        return DiceLoss()
    if name == "bce":
        pw = torch.tensor([pos_weight]) if pos_weight else None
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    raise ValueError(f"Unknown loss: {name}")
