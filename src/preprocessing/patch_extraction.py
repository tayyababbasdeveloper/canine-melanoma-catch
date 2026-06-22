"""Patch extraction.

Whole-slide images are far too large to feed into a CNN directly, so we tile each
slide into fixed-size patches and keep only those containing enough tissue
(proposal section 3.1.1: multi-magnification patch extraction).

This module provides three layers:
  * ``extract_patches``            — image-only tiling (Week 1-2 classification).
  * ``extract_patches_with_mask``  — paired image+mask tiling for segmentation.
  * ``extract_multimag_*``         — the same, repeated at several magnifications
                                     (5x / 10x / 20x) by downsampling the slide.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

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


# =====================================================================
# Segmentation: paired image + mask tiling
# =====================================================================
def extract_patches_with_mask(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_fraction: float = 0.5,
    max_patches: int = 2000,
    tissue_img: np.ndarray | None = None,
):
    """Yield (row, col, image_patch, mask_patch) for tissue-bearing tiles.

    ``mask`` is a 2-D array the same H×W as the slide (ground-truth tumour
    annotation). The tissue test uses the image (not the mask) so that
    tumour-free tissue patches are also retained as negative examples — both
    are needed to train an unbiased segmentation model.

    ``tissue_img`` lets the tissue mask be computed from a DIFFERENT image than
    the one whose pixels are saved — typically the ORIGINAL slide, while
    ``img_rgb`` is the stain-normalised slide. Where tissue is located is a
    content decision that must not depend on colour normalisation, so detecting
    it on the original image is both more robust and more correct.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    tmask = tissue_mask(tissue_img if tissue_img is not None else img_rgb)
    h, w = img_rgb.shape[:2]
    count = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            tile_tissue = tmask[y:y + patch_size, x:x + patch_size]
            if tile_tissue.mean() < min_tissue_fraction:
                continue
            patch = img_rgb[y:y + patch_size, x:x + patch_size]
            mpatch = mask[y:y + patch_size, x:x + patch_size]
            yield y, x, patch, mpatch
            count += 1
            if count >= max_patches:
                return


def _downsample_for_magnification(arr: np.ndarray, factor: float, is_mask: bool):
    """Downsample an array by ``factor`` (>=1). Nearest for masks, area for images."""
    if factor <= 1.0:
        return arr
    h, w = arr.shape[:2]
    new_w, new_h = max(1, int(round(w / factor))), max(1, int(round(h / factor)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(arr, (new_w, new_h), interpolation=interp)


def extract_multimag_with_mask(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    magnifications,
    base_magnification: int,
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_fraction: float = 0.5,
    max_patches: int = 2000,
    tissue_img: np.ndarray | None = None,
):
    """Extract paired image+mask patches at several magnifications.

    A lower target magnification (e.g. 5x from a 20x slide) is produced by
    downsampling the slide by ``base / target`` before tiling, so each patch
    covers a wider field of view at lower detail — exactly the multi-scale
    context the proposal calls for (section 3.1.1).

    ``tissue_img`` (optional) is the image used for tissue detection (e.g. the
    original, pre-normalisation slide); it is downsampled alongside ``img_rgb``.

    Yields (magnification, row, col, image_patch, mask_patch).
    """
    for mag in magnifications:
        factor = base_magnification / float(mag)
        img_m = _downsample_for_magnification(img_rgb, factor, is_mask=False)
        mask_m = _downsample_for_magnification(mask, factor, is_mask=True)
        tissue_m = (None if tissue_img is None else
                    _downsample_for_magnification(tissue_img, factor, is_mask=False))
        for (y, x, patch, mpatch) in extract_patches_with_mask(
            img_m, mask_m, patch_size, stride, min_tissue_fraction, max_patches,
            tissue_img=tissue_m,
        ):
            yield mag, y, x, patch, mpatch


def save_mask(path, mask: np.ndarray) -> None:
    """Save a binary/label mask as an 8-bit PNG (0 / 255)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    cv2.imwrite(str(path), m.astype(np.uint8))
