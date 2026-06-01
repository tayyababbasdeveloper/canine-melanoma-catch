"""Macenko (2009) stain normalisation.

Histopathology slides vary in colour because of differences in staining, scanners
and labs. Macenko normalisation maps every slide onto a common H&E colour
appearance, which is essential before training CNNs (proposal section 3.1.1).

Reference:
    Macenko, M. et al. (2009) 'A method for normalizing histology slides for
    quantitative analysis', IEEE ISBI, pp. 1107-1110.

The implementation works in optical-density (OD) space:
    OD = -log10(I / Io)
estimates the two stain vectors (haematoxylin & eosin) as the extremes of the
OD plane, then re-projects the slide onto a fixed reference stain matrix.
"""
from __future__ import annotations

import numpy as np

# Reference H&E stain matrix and max concentrations (Macenko defaults).
# Columns = haematoxylin, eosin; rows = R, G, B optical densities.
HE_REF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]])
MAX_C_REF = np.array([1.9705, 1.0308])


class MacenkoNormalizer:
    def __init__(self, io: int = 240, alpha: float = 1.0, beta: float = 0.15):
        self.io = io        # background light intensity
        self.alpha = alpha  # robust percentile for the stain-angle extremes
        self.beta = beta    # OD threshold to discard transparent pixels

    # ---- internal helpers -------------------------------------------------
    def _rgb_to_od(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB [0,255] to optical density, flattened to (N, 3)."""
        img = img.astype(np.float64).reshape((-1, 3))
        return -np.log((img + 1.0) / self.io)

    def _stain_matrix(self, od: np.ndarray) -> np.ndarray:
        """Estimate the (3, 2) H&E stain matrix from OD pixels."""
        od_hat = od[~np.any(od < self.beta, axis=1)]  # drop transparent pixels
        if od_hat.shape[0] < 10:
            raise ValueError("Not enough tissue pixels for stain estimation.")

        # Plane spanned by the two largest eigenvectors of the OD covariance
        _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        plane = eigvecs[:, 1:3]
        proj = od_hat.dot(plane)

        angles = np.arctan2(proj[:, 1], proj[:, 0])
        min_a = np.percentile(angles, self.alpha)
        max_a = np.percentile(angles, 100 - self.alpha)

        v_min = plane.dot(np.array([np.cos(min_a), np.sin(min_a)]))
        v_max = plane.dot(np.array([np.cos(max_a), np.sin(max_a)]))

        # Order so haematoxylin (first) has the larger red OD
        if v_min[0] > v_max[0]:
            he = np.array([v_min, v_max]).T
        else:
            he = np.array([v_max, v_min]).T
        return he

    # ---- public API -------------------------------------------------------
    def normalize(self, img_rgb: np.ndarray) -> np.ndarray:
        """Return the stain-normalised RGB uint8 image."""
        h, w = img_rgb.shape[:2]
        od = self._rgb_to_od(img_rgb)
        he = self._stain_matrix(od)

        # Stain concentrations for every pixel: C = HE^+ . OD
        concentrations = np.linalg.lstsq(he, od.T, rcond=None)[0]
        max_c = np.percentile(concentrations, 99, axis=1)
        concentrations *= (MAX_C_REF / max_c)[:, np.newaxis]

        # Reconstruct using the fixed reference stain matrix
        norm = self.io * np.exp(-HE_REF.dot(concentrations))
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        return norm.T.reshape(h, w, 3)
