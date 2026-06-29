"""Synthetic multi-class tumour slides (DEMO ONLY) for classification.

The proposal's classification task (section 3.1.3) is multi-class tumour-subtype
recognition. The real CATCH dataset provides subtype labels; until it is
downloaded, this module fabricates slides for a few subtypes, each with a
DISTINCT visual signature so a CNN can actually learn to tell them apart:

  * melanocytic     — dense, dark melanin pigment clusters
  * mast_cell       — medium purple cells with fine granular speckles
  * squamous_cell   — large pale-pink polygonal cells in sheets, sparse nuclei

A near-white glass background and a reliably-filled pink tissue body are added so
that, after tiling, every slide yields a comparable number of tissue patches.

THIS IS NOT REAL DATA — it only exercises the classification pipeline end-to-end.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

CLASSES = ("melanocytic", "mast_cell", "squamous_cell")


def _tissue_body(img, size, rng):
    base = np.array([200, 140, 180])
    margin = size // 10
    cv2.ellipse(img, (size // 2, size // 2),
                (size // 2 - margin, size // 2 - margin), 0, 0, 360, base.tolist(), -1)
    for _ in range(35):
        cx, cy = rng.integers(margin, size - margin, 2)
        r = rng.integers(size // 8, size // 4)
        cv2.circle(img, (int(cx), int(cy)), int(r), (base + rng.normal(0, 10, 3)).tolist(), -1)
    for _ in range(4):
        cx, cy = rng.integers(0, size, 2)
        cv2.circle(img, (int(cx), int(cy)), int(rng.integers(size // 16, size // 10)),
                   (245, 245, 245), -1)


def generate_class_slide(label: str, size: int = 1536, seed: int = 0) -> np.ndarray:
    """Generate one synthetic H&E slide whose cells match the given subtype."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 245, dtype=np.float64)
    _tissue_body(img, size, rng)

    if label == "melanocytic":
        # dense dark melanin clusters
        for _ in range(6000):
            cx, cy = rng.integers(0, size, 2)
            r = rng.integers(2, 6)
            colour = np.array([60, 40, 55]) + rng.normal(0, 16, 3)
            cv2.circle(img, (int(cx), int(cy)), int(r), colour.tolist(), -1)

    elif label == "mast_cell":
        # medium purple round cells with fine granular speckles
        for _ in range(2500):
            cx, cy = rng.integers(0, size, 2)
            r = rng.integers(4, 8)
            cv2.circle(img, (int(cx), int(cy)), int(r),
                       (np.array([120, 80, 160]) + rng.normal(0, 12, 3)).tolist(), -1)
        for _ in range(9000):  # granules
            cx, cy = rng.integers(0, size, 2)
            cv2.circle(img, (int(cx), int(cy)), 1,
                       (np.array([90, 50, 120]) + rng.normal(0, 15, 3)).tolist(), -1)

    else:  # squamous_cell — large pale polygonal cells, sparse nuclei
        for _ in range(1200):
            cx, cy = rng.integers(0, size, 2)
            r = rng.integers(10, 20)
            cv2.circle(img, (int(cx), int(cy)), int(r),
                       (np.array([225, 180, 200]) + rng.normal(0, 8, 3)).tolist(), -1)
        for _ in range(900):  # sparse small nuclei
            cx, cy = rng.integers(0, size, 2)
            cv2.circle(img, (int(cx), int(cy)), int(rng.integers(2, 5)),
                       (np.array([110, 70, 140]) + rng.normal(0, 15, 3)).tolist(), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_demo_classification_slides(raw_dir, slides_per_class: int = 4,
                                    size: int = 1536) -> list[tuple[Path, str]]:
    """Write synthetic slides for each subtype. Returns (path, label) pairs."""
    raw_dir = Path(raw_dir) / "cls_images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    casts = [(0.9, 1.0, 1.1), (1.1, 0.95, 0.95), (0.95, 1.05, 1.0), (1.05, 0.98, 1.02)]
    pairs = []
    for ci, label in enumerate(CLASSES):
        for k in range(slides_per_class):
            img = generate_class_slide(label, size=size, seed=200 + ci * 10 + k)
            cast = casts[k % len(casts)]
            img = np.clip(img.astype(np.float64) * np.array(cast), 0, 255).astype(np.uint8)
            p = raw_dir / f"{label}_{k:02d}.png"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            pairs.append((p, label))
    return pairs
