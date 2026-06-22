"""Test-set evaluation and qualitative visualisation (proposal section 3.1.4).

Computes Dice / IoU / pixel-accuracy / Hausdorff over the held-out test split and
renders prediction-overlay panels (image | ground-truth | prediction) so the
segmentation quality can be inspected visually in the supervisor meeting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.models.metrics import SegMetrics


def _denormalise(img_t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation -> uint8 RGB for display."""
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Aggregate test metrics over the loader."""
    model.eval()
    metrics = SegMetrics()
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        metrics.update(logits, masks)
    return metrics.compute()


@torch.no_grad()
def save_prediction_grid(model, dataset, device, out_path: Path,
                         n: int = 6, title: str = "U-Net predictions"):
    """Save an (image | ground truth | prediction) panel for n samples."""
    model.eval()
    n = min(n, len(dataset))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        image, mask = dataset[i]
        logits = model(image.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5)

        axes[i, 0].imshow(_denormalise(image)); axes[i, 0].set_title("Image")
        axes[i, 1].imshow(mask[0].cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(pred, cmap="gray"); axes[i, 2].set_title("Prediction")
        for j in range(3):
            axes[i, j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_training_curves(history: list[dict], out_path: Path,
                         title: str = "U-Net training"):
    """Plot train/val loss and val Dice across epochs."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_dice"] for h in history], "g--", label="val Dice")
    ax2.set_ylabel("Dice"); ax2.legend(loc="lower right")

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
