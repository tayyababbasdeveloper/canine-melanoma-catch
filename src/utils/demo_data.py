"""Synthetic H&E slide generator (DEMO ONLY).

Lets the full Week 1-2 pipeline run end-to-end without the real CATCH dataset, so
the supervisor can be shown genuine outputs (stain normalisation before/after,
extracted patches). Each synthetic slide simulates H&E-stained tissue with a
deliberate colour cast, so Macenko normalisation has a visible effect.

THIS IS NOT REAL DATA — it is only for demonstrating the pipeline mechanics.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2


def generate_he_slide(
    size: int = 1024,
    n_cells: int = 1400,
    color_cast: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 0,
) -> np.ndarray:
    """Generate one synthetic H&E-like RGB image with a given colour cast."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 245, dtype=np.float64)  # near-white glass

    # Broad pink eosin "tissue" body covering most of the slide
    base_tissue = np.array([200, 140, 180])
    for _ in range(60):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(size // 6, size // 2)
        colour = base_tissue + rng.normal(0, 10, 3)
        cv2.circle(img, (int(cx), int(cy)), int(r), colour.tolist(), -1)

    # A few white "background glass" gaps for realism
    for _ in range(6):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(size // 12, size // 6)
        cv2.circle(img, (int(cx), int(cy)), int(r), (245, 245, 245), -1)

    # Purple haematoxylin "nuclei" (small dark dots)
    for _ in range(n_cells):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(3, 8)
        colour = np.array([110, 70, 150]) + rng.normal(0, 15, 3)
        cv2.circle(img, (int(cx), int(cy)), int(r), colour.tolist(), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply a per-channel colour cast (simulates scanner/lab variation)
    img *= np.array(color_cast)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_demo_slides(raw_dir: str | Path) -> list[Path]:
    """Write two demo slides with DIFFERENT colour casts into raw_dir.

    Two casts demonstrate why normalisation is needed: the pipeline should map
    both onto a common colour appearance.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("demo_slide_bluish.png", (0.82, 0.88, 1.12), 1),   # bluish cast
        ("demo_slide_pinkish.png", (1.12, 0.92, 0.95), 2),  # pinkish cast
    ]
    paths = []
    for fname, cast, seed in specs:
        img = generate_he_slide(color_cast=cast, seed=seed)
        out = raw_dir / fname
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(out)
    return paths
