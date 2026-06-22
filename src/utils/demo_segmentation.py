"""Synthetic H&E slides WITH tumour masks (DEMO ONLY).

Week 1-2 produced synthetic slides for the classification/preprocessing demo.
The Week 3-4 segmentation work additionally needs *ground-truth tumour masks*,
which the real CATCH annotations provide but the synthetic demo must fabricate.

Each generated slide contains:
  * a broad pink eosin "tissue" body on near-white glass,
  * scattered purple haematoxylin "nuclei" (normal tissue),
  * one or more irregular TUMOUR regions of densely packed, darkly pigmented
    (melanin-like) cells — the hallmark this segmentation task must learn.

The returned mask marks exactly those tumour regions, giving the U-Net a
genuine (if synthetic) image->mask mapping so Dice/IoU are meaningful.

THIS IS NOT REAL DATA — it only exercises the segmentation pipeline end-to-end
until the annotated CATCH slides are downloaded.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2


def _irregular_blob(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a boolean mask with one small, irregular blob (jittered circles).

    Tumours are kept deliberately small relative to the slide so that the patch
    dataset contains a realistic MIX of tumour and tumour-free tissue tiles —
    otherwise the segmentation task degenerates into "predict all foreground".
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    margin = size // 6
    cx, cy = rng.integers(margin, size - margin, 2)
    base_r = rng.integers(size // 16, size // 9)
    for _ in range(rng.integers(4, 8)):
        ox, oy = rng.integers(-base_r // 2, base_r // 2 + 1, 2)
        r = rng.integers(base_r // 2, base_r)
        cv2.circle(mask, (int(cx + ox), int(cy + oy)), int(r), 255, -1)
    return mask > 0


def generate_he_slide_with_mask(
    size: int = 2048,
    n_cells: int = 5000,
    color_cast: tuple[float, float, float] = (1.0, 1.0, 1.0),
    n_tumours: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic H&E RGB image and its binary tumour mask."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 245, dtype=np.float64)  # near-white glass
    base_tissue = np.array([200, 140, 180])

    # Reliable broad eosin "tissue" body: a large central ellipse covering most
    # of the slide so that, after tiling, EVERY slide yields a comparable number
    # of tissue patches (consistent dataset size across slides).
    margin = size // 10
    cv2.ellipse(img, (size // 2, size // 2),
                (size // 2 - margin, size // 2 - margin),
                0, 0, 360, base_tissue.tolist(), -1)
    # texture on top of the body
    for _ in range(40):
        cx, cy = rng.integers(margin, size - margin, 2)
        r = rng.integers(size // 8, size // 4)
        colour = base_tissue + rng.normal(0, 10, 3)
        cv2.circle(img, (int(cx), int(cy)), int(r), colour.tolist(), -1)

    # A few small white "background glass" gaps for realism (kept modest so they
    # do not dominate any tile)
    for _ in range(5):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(size // 16, size // 10)
        cv2.circle(img, (int(cx), int(cy)), int(r), (245, 245, 245), -1)

    # Normal, sparse purple haematoxylin "nuclei"
    for _ in range(n_cells):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(3, 7)
        colour = np.array([110, 70, 150]) + rng.normal(0, 15, 3)
        cv2.circle(img, (int(cx), int(cy)), int(r), colour.tolist(), -1)

    # ---- Tumour regions: dense, darkly pigmented (melanin-like) cells ----
    mask = np.zeros((size, size), dtype=np.uint8)
    for t in range(n_tumours):
        blob = _irregular_blob(size, rng)
        mask[blob] = 1
        ys, xs = np.where(blob)
        if len(xs) == 0:
            continue
        # pack many dark melanin dots inside the blob
        n_dense = int(0.06 * len(xs))
        idx = rng.choice(len(xs), size=max(1, n_dense), replace=True)
        for j in idx:
            r = rng.integers(2, 6)
            colour = np.array([70, 45, 60]) + rng.normal(0, 18, 3)  # dark brown/purple
            cv2.circle(img, (int(xs[j]), int(ys[j])), int(r), colour.tolist(), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img *= np.array(color_cast)                      # scanner/lab colour variation
    img = np.clip(img, 0, 255).astype(np.uint8)

    # smooth the mask boundary slightly (realistic annotation)
    mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = (mask > 0).astype(np.uint8)
    return img, mask


def make_demo_segmentation_slides(
    raw_dir: str | Path,
    n_slides: int = 8,
    size: int = 2048,
) -> list[tuple[Path, Path]]:
    """Write ``n_slides`` synthetic slides + masks. Returns (image, mask) paths.

    Colour casts are varied per slide so stain normalisation has a visible effect
    and the model cannot cheat on colour alone.
    """
    raw_dir = Path(raw_dir)
    img_dir = raw_dir / "images"
    msk_dir = raw_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    casts = [(0.82, 0.88, 1.12), (1.12, 0.92, 0.95), (0.95, 1.08, 0.98),
             (1.05, 0.95, 1.05), (0.90, 1.0, 1.10), (1.10, 1.0, 0.90),
             (0.88, 1.05, 1.0), (1.0, 0.9, 1.08)]
    pairs: list[tuple[Path, Path]] = []
    for i in range(n_slides):
        cast = casts[i % len(casts)]
        n_t = 2 + (i % 3)  # 2-4 tumour regions
        img, mask = generate_he_slide_with_mask(
            size=size, color_cast=cast, n_tumours=n_t, seed=100 + i,
        )
        slide_id = f"demo_seg_{i:02d}"
        ip = img_dir / f"{slide_id}.png"
        mp = msk_dir / f"{slide_id}.png"
        cv2.imwrite(str(ip), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mp), mask * 255)
        pairs.append((ip, mp))
    return pairs
