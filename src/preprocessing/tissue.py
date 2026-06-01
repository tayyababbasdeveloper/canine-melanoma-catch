"""Tissue detection.

Histopathology slides are mostly white/empty glass. We detect tissue using the
saturation channel of HSV (tissue is coloured by H&E stains; background glass is
near-grey/white with low saturation), combined with Otsu thresholding.
"""
from __future__ import annotations

import numpy as np
import cv2


def tissue_mask(img_rgb: np.ndarray, sat_threshold: int | None = None) -> np.ndarray:
    """Return a boolean mask (True = tissue) for an RGB image."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    if sat_threshold is None:
        # Otsu picks the threshold automatically from the saturation histogram
        sat_threshold, _ = cv2.threshold(
            saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    mask = saturation > sat_threshold

    # Clean up speckle with a morphological opening
    mask_u8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8 > 0


def tissue_fraction(img_rgb: np.ndarray) -> float:
    """Fraction of the image covered by tissue (0.0 - 1.0)."""
    mask = tissue_mask(img_rgb)
    return float(mask.mean())
