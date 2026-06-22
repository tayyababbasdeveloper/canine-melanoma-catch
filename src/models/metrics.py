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
    """Accumulate per-batch metrics and report the mean over an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._dice, self._iou, self._acc, self._hd, self._n = 0.0, 0.0, 0.0, [], 0

    @torch.no_grad()
    def update(self, logits, targets):
        """Update with a batch; metrics are averaged per-sample."""
        b = logits.shape[0]
        for i in range(b):
            lo, ta = logits[i:i + 1], targets[i:i + 1]
            self._dice += dice_coefficient(lo, ta)
            self._iou += iou_score(lo, ta)
            self._acc += pixel_accuracy(lo, ta)
            self._hd.append(hausdorff_distance(lo, ta))
            self._n += 1

    def compute(self) -> dict:
        n = max(self._n, 1)
        hd = np.array(self._hd, dtype=np.float64)
        return {
            "dice": self._dice / n,
            "iou": self._iou / n,
            "pixel_acc": self._acc / n,
            "hausdorff": float(np.nanmean(hd)) if np.isfinite(hd).any() else float("nan"),
        }
