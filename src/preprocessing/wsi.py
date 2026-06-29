"""Whole-slide image (WSI) reading for the real CATCH dataset.

The CATCH slides are pyramidal Aperio ``.svs`` files (level 0 = 0.25 um/px ~ 40x)
that are far too large to load into memory whole (a single slide can be tens of
thousands of pixels on a side). This module reads them **tile by tile** at a
chosen objective magnification using OpenSlide, so patch extraction scales to the
full 350-slide collection without ever materialising a whole slide.

It deliberately mirrors the patch geometry used by the synthetic demo
(``patch_size`` tiles on a regular grid) so the *same* downstream code — tissue
detection, Macenko, the segmentation/classification datasets — works unchanged on
real and demo data.

OpenSlide (and its binaries) are imported lazily, so the demo pipeline still runs
on a machine without OpenSlide installed.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

WSI_EXTENSIONS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")


def _require_openslide():
    try:
        import openslide  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openslide-python and the OpenSlide binaries are required to read "
            "whole-slide images. Install with `pip install openslide-python` and "
            "the platform binaries from https://openslide.org/download/ (on "
            "Windows, add the OpenSlide `bin` folder to PATH)."
        ) from exc
    return openslide


def open_slide(path: str | Path):
    """Open a WSI with OpenSlide (lazy import)."""
    openslide = _require_openslide()
    return openslide.OpenSlide(str(path))


def slide_magnification(slide, default: float = 40.0) -> float:
    """Objective magnification of level 0, from slide metadata.

    Falls back to ``default`` (CATCH scans at 40x / 0.25 um/px) when the property
    is missing.
    """
    import openslide
    val = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _best_level(slide, downsample: float) -> int:
    """Pick the highest-resolution level whose downsample does not exceed the
    requested one (so the final resize is always a downscale, never an upscale)."""
    best, best_ds = 0, 1.0
    for lvl, ds in enumerate(slide.level_downsamples):
        if ds <= downsample + 1e-3 and ds >= best_ds:
            best, best_ds = lvl, ds
    return best


def iter_tiles(
    slide,
    patch_size: int = 256,
    read_magnification: float = 20.0,
    base_magnification: float | None = None,
    stride: int | None = None,
):
    """Yield ``(x0, y0, patch_rgb)`` tiles across the slide.

    ``x0, y0`` are top-left coordinates in **level-0 pixels** (so annotations,
    which are stored in level-0 coordinates, can be rasterised against them).
    ``patch_rgb`` is a ``patch_size x patch_size x 3`` uint8 RGB array sampled at
    ``read_magnification``.
    """
    stride = stride or patch_size
    base = base_magnification or slide_magnification(slide)
    downsample = base / float(read_magnification)          # e.g. 40 / 20 = 2.0
    level = _best_level(slide, downsample)
    level_ds = slide.level_downsamples[level]
    # how many level-0 pixels one output patch spans
    span0 = int(round(patch_size * downsample))
    stride0 = int(round(stride * downsample))
    # size to read at the chosen level before the final exact resize to patch_size
    read_at_level = max(1, int(round(span0 / level_ds)))

    w0, h0 = slide.dimensions
    import cv2
    for y0 in range(0, h0 - span0 + 1, stride0):
        for x0 in range(0, w0 - span0 + 1, stride0):
            region = slide.read_region((x0, y0), level, (read_at_level, read_at_level))
            arr = np.asarray(region.convert("RGB"))
            if arr.shape[0] != patch_size or arr.shape[1] != patch_size:
                arr = cv2.resize(arr, (patch_size, patch_size),
                                 interpolation=cv2.INTER_AREA)
            yield x0, y0, arr


def read_thumbnail(slide, max_size: int = 4096) -> np.ndarray:
    """Low-resolution RGB thumbnail of the whole slide (for QA / overview)."""
    thumb = slide.get_thumbnail((max_size, max_size))
    return np.array(thumb.convert("RGB"))


def tile_geometry(slide, patch_size: int, read_magnification: float,
                  base_magnification: float | None = None) -> dict:
    """Report the level-0 span and downsample for a tile (used by the mask
    rasteriser so it shares the exact geometry of :func:`iter_tiles`)."""
    base = base_magnification or slide_magnification(slide)
    downsample = base / float(read_magnification)
    return {
        "downsample": downsample,
        "span0": int(round(patch_size * downsample)),
        "base_magnification": base,
        "level0_dimensions": slide.dimensions,
    }
