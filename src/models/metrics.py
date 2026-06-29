"""Segmentation evaluation metrics (proposal section 3.1.4).

Dice Similarity Coefficient (DSC), Intersection over Union (IoU) and pixel
accuracy are computed on binarised predictions. Hausdorff distance measures the
worst-case boundary disagreement and is computed on mask contours.

All functions accept either torch tensors or numpy arrays of shape (..., H, W)
or (..., 1, H, W); logits are thresholded at 0.5 after a sigmoid.
"""
from __future__ import annotations

import numpy as np
import torch


def _to_binary_np(x, threshold: float = 0.5, is_logit: bool = True) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float()
        if is_logit:
            x = torch.sigmoid(x)
        x = x.numpy()
    else:
        x = np.asarray(x, dtype=np.float32)
    return (x >= threshold).astype(np.uint8)


@torch.no_grad()
def dice_coefficient(logits, targets, eps: float = 1e-7) -> float:
    pred = _to_binary_np(logits, is_logit=True)
    gt = _to_binary_np(targets, is_logit=False)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * inter + eps) / (denom + eps))


@torch.no_grad()
def iou_score(logits, targets, eps: float = 1e-7) -> float:
    pred = _to_binary_np(logits, is_logit=True)
    gt = _to_binary_np(targets, is_logit=False)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


@torch.no_grad()
def pixel_accuracy(logits, targets) -> float:
    pred = _to_binary_np(logits, is_logit=True)
    gt = _to_binary_np(targets, is_logit=False)
    return float((pred == gt).mean())


@torch.no_grad()
def hausdorff_distance(logits, targets) -> float:
    """Symmetric Hausdorff distance between mask boundaries (single image).

    Returns NaN when one mask is empty (distance undefined). Use ``nanmean`` to
    aggregate so empty-mask patches do not corrupt the average.
    """
    from scipy.spatial.distance import directed_hausdorff

    pred = _to_binary_np(logits, is_logit=True).squeeze()
    gt = _to_binary_np(targets, is_logit=False).squeeze()
    pts_p = np.argwhere(pred > 0)
    pts_g = np.argwhere(gt > 0)
    if len(pts_p) == 0 or len(pts_g) == 0:
        return float("nan")
    return float(max(directed_hausdorff(pts_p, pts_g)[0],
                     directed_hausdorff(pts_g, pts_p)[0]))


class SegMetrics:
    """Accumulate metrics over an epoch.

    Dice and IoU are **micro-averaged** (pixel-aggregated across all patches):
    intersection / union are summed over every pixel of every patch, then a
    single Dice/IoU is computed. This is the honest metric for a patch dataset
    with many background-only patches — unlike mean-per-patch Dice, a model that
    simply predicts "empty everywhere" scores ~0 (not ~1), because the few
    tumour patches dominate the aggregate. Pixel-accuracy is micro too; Hausdorff
    is averaged over tumour-containing patches only.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._inter = 0.0
        self._pred = 0.0
        self._gt = 0.0
        self._union = 0.0
        self._correct = 0.0
        self._pixels = 0.0
        self._hd = []

    @torch.no_grad()
    def update(self, logits, targets):
        pred = _to_binary_np(logits, is_logit=True).astype(np.float64)
        gt = _to_binary_np(targets, is_logit=False).astype(np.float64)
        self._inter += float((pred * gt).sum())
        self._pred += float(pred.sum())
        self._gt += float(gt.sum())
        self._union += float(np.logical_or(pred, gt).sum())
        self._correct += float((pred == gt).sum())
        self._pixels += float(pred.size)
        # Hausdorff still makes sense only per tumour-containing image
        for i in range(logits.shape[0]):
            self._hd.append(hausdorff_distance(logits[i:i + 1], targets[i:i + 1]))

    def compute(self, eps: float = 1e-7) -> dict:
        hd = np.array(self._hd, dtype=np.float64)
        return {
            "dice": (2 * self._inter + eps) / (self._pred + self._gt + eps),
            "iou": (self._inter + eps) / (self._union + eps),
            "pixel_acc": self._correct / max(self._pixels, 1),
            "hausdorff": float(np.nanmean(hd)) if np.isfinite(hd).any() else float("nan"),
        }
