"""Quality assessment of whole-slide images.

For every slide we compute objective quality metrics and flag slides that should
be excluded (with a documented reason), as required by the proposal's data-quality
risk mitigation. Produces a CSV report in outputs/logs/.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
import pandas as pd

from src.preprocessing.image_io import load_image_rgb, WSI_EXTENSIONS, STD_EXTENSIONS
from src.preprocessing.tissue import tissue_fraction


def blur_score(img_rgb: np.ndarray) -> float:
    """Variance of the Laplacian — low values mean a blurry/out-of-focus slide."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_image(img_rgb: np.ndarray, q_cfg: dict) -> dict:
    """Compute quality metrics for a single image and decide pass/flag."""
    h, w = img_rgb.shape[:2]
    brightness = float(img_rgb.mean())
    blur = blur_score(img_rgb)
    tissue = tissue_fraction(img_rgb)

    reasons = []
    if w < q_cfg["min_width"] or h < q_cfg["min_height"]:
        reasons.append("too_small")
    if blur < q_cfg["blur_laplacian_min"]:
        reasons.append("blurry")
    if tissue < q_cfg["min_tissue_fraction"]:
        reasons.append("insufficient_tissue")
    if brightness < q_cfg["brightness_low"]:
        reasons.append("too_dark")
    if brightness > q_cfg["brightness_high"]:
        reasons.append("washed_out")

    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "blur_score": round(blur, 2),
        "tissue_fraction": round(tissue, 4),
        "passed": len(reasons) == 0,
        "flag_reason": ";".join(reasons) if reasons else "",
    }


def assess_directory(input_dir: Path, q_cfg: dict, logger) -> pd.DataFrame:
    """Run quality assessment over every slide/image in a directory."""
    exts = set(WSI_EXTENSIONS) | set(STD_EXTENSIONS)
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in exts)

    rows = []
    for f in files:
        try:
            img = load_image_rgb(f)
            result = assess_image(img, q_cfg)
            result["file"] = f.name
            rows.append(result)
            status = "PASS" if result["passed"] else f"FLAG ({result['flag_reason']})"
            logger.info("QA %-30s -> %s", f.name, status)
        except Exception as exc:  # corrupt / unreadable slide
            rows.append({"file": f.name, "passed": False,
                         "flag_reason": f"unreadable:{exc}"})
            logger.warning("QA %-30s -> UNREADABLE (%s)", f.name, exc)

    df = pd.DataFrame(rows)
    cols = ["file", "width", "height", "brightness", "blur_score",
            "tissue_fraction", "passed", "flag_reason"]
    return df.reindex(columns=[c for c in cols if c in df.columns])
