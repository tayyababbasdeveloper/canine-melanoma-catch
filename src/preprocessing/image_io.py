"""Image loading utilities.

Reads either standard images (PNG/JPG/TIFF) for development/demo, or whole-slide
images (.svs/.ndpi) via OpenSlide when available. WSIs are loaded at a downsampled
thumbnail level so they fit in memory for quality assessment and tiling.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

WSI_EXTENSIONS = (".svs", ".ndpi", ".mrxs")
STD_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_image_rgb(path: str | Path, max_size: int = 4096) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array.

    For huge whole-slide images, an OpenSlide thumbnail is returned (<= max_size).
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in WSI_EXTENSIONS:
        return _load_wsi_thumbnail(path, max_size)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _downscale(img, max_size)


def _load_wsi_thumbnail(path: Path, max_size: int) -> np.ndarray:
    try:
        import openslide  # imported lazily so the demo works without it
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openslide-python is required to read whole-slide images. "
            "Install it and the OpenSlide binaries (https://openslide.org/download/)."
        ) from exc

    slide = openslide.OpenSlide(str(path))
    thumb = slide.get_thumbnail((max_size, max_size))
    return np.array(thumb.convert("RGB"))


def _downscale(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def save_rgb(path: str | Path, img_rgb: np.ndarray) -> None:
    """Save an RGB uint8 array to disk (handles RGB->BGR for OpenCV)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
