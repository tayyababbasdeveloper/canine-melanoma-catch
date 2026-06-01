"""Patch extraction.

Whole-slide images are far too large to feed into a CNN directly, so we tile each
slide into fixed-size patches and keep only those containing enough tissue
(proposal section 3.1.1: multi-magnification patch extraction).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.preprocessing.tissue import tissue_mask
from src.preprocessing.image_io import save_rgb


def extract_patches(
    img_rgb: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_fraction: float = 0.5,
    max_patches: int = 2000,
):
    """Yield (row, col, patch) tuples for tissue-bearing tiles of the image."""
    mask = tissue_mask(img_rgb)
    h, w = img_rgb.shape[:2]
    count = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_mask = mask[y:y + patch_size, x:x + patch_size]
            if patch_mask.mean() < min_tissue_fraction:
                continue
            patch = img_rgb[y:y + patch_size, x:x + patch_size]
            yield y, x, patch
            count += 1
            if count >= max_patches:
                return


def extract_and_save(
    img_rgb: np.ndarray,
    out_dir: Path,
    slide_id: str,
    p_cfg: dict,
) -> int:
    """Extract patches from one slide and write them to out_dir. Returns count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for (y, x, patch) in extract_patches(
        img_rgb,
        patch_size=p_cfg["patch_size"],
        stride=p_cfg["stride"],
        min_tissue_fraction=p_cfg["min_tissue_fraction"],
        max_patches=p_cfg["max_patches_per_slide"],
    ):
        fname = f"{slide_id}_y{y}_x{x}.png"
        save_rgb(out_dir / fname, patch)
        n += 1
    return n
