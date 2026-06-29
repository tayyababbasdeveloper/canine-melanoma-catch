"""Week 1-2 preprocessing pipeline — single entry point.

Runs the full Foundation-phase data pipeline and writes presentation-ready
figures and reports to outputs/ for the supervisor meeting:

    1. (optional) generate synthetic demo slides
    2. quality assessment      -> outputs/logs/quality_report.csv
    3. Macenko stain norm.      -> outputs/figures/stain_normalization_*.png
    4. patch extraction         -> data/processed/patches/ + grid figure
    5. stratified 70/15/15 split-> data/processed/splits/*.csv + summary

Usage:
    python scripts/run_week1_2_pipeline.py --demo          # synthetic slides
    python scripts/run_week1_2_pipeline.py --input data/raw # real CATCH slides
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

# Make 'src' importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger
from src.utils.demo_data import make_demo_slides
from src.preprocessing.image_io import load_image_rgb, save_rgb, WSI_EXTENSIONS, STD_EXTENSIONS
from src.data_acquisition.quality_check import assess_image
from src.preprocessing.stain_normalization import MacenkoNormalizer
from src.preprocessing.patch_extraction import extract_patches, extract_and_save
from src.preprocessing.dataset_split import make_split


def list_slides(input_dir: Path) -> list[Path]:
    exts = set(WSI_EXTENSIONS) | set(STD_EXTENSIONS)
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in exts)


def slide_label(slide: Path, subtypes: set[str]) -> str:
    """Subtype label for a slide from its REAL source.

    Order of precedence:
      1. a CATCH subtype parent folder (real layout: data/raw/<Subtype>/slide.svs)
      2. the ``<label>_<n>`` filename convention used by labelled PNG slides
      3. for the two synthetic Week 1-2 demo slides (which have NO real subtype),
         an explicit ``demo_*`` placeholder — never a biological label.
    """
    for parent in slide.parents:
        if parent.name in subtypes:
            return parent.name
    stem = slide.stem
    if stem.startswith("demo_slide"):
        return "demo_bluish" if "blu" in stem else "demo_pinkish"
    if "_" in stem:
        cand = stem.rsplit("_", 1)[0]
        if cand:
            return cand
    return "unknown"


def save_before_after(before, after, out_path, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(before); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(after);  ax[1].set_title("Macenko normalised"); ax[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_patch_grid(patches, out_path, title, n=16):
    n = min(n, len(patches))
    if n == 0:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < n:
            ax.imshow(patches[i]); ax.axis("off")
        else:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Week 1-2 CATCH preprocessing pipeline")
    parser.add_argument("--demo", action="store_true",
                        help="generate synthetic slides instead of using real data")
    parser.add_argument("--input", default=None, help="directory of slides")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    logger = get_logger("pipeline", cfg["paths"]["logs_dir"])

    figures_dir = Path(cfg["paths"]["figures_dir"])
    logs_dir = Path(cfg["paths"]["logs_dir"])

    # ---- 1. choose input slides ----
    if args.demo:
        logger.info("DEMO mode: generating synthetic H&E slides")
        slides = make_demo_slides(cfg["paths"]["raw_dir"])
    else:
        input_dir = Path(args.input) if args.input else Path(cfg["paths"]["raw_dir"])
        slides = list_slides(input_dir)
    if not slides:
        logger.error("No slides found. Use --demo or place slides in data/raw/.")
        return
    logger.info("Processing %d slide(s)", len(slides))

    normalizer = MacenkoNormalizer(**cfg["stain_normalization"])
    patches_root = Path(cfg["paths"]["patches_dir"])
    subtypes = set(cfg["catch"]["subtypes"])
    qa_rows, split_rows = [], []

    # ---- 2-4. per-slide processing ----
    for slide in slides:
        slide_id = slide.stem
        img = load_image_rgb(slide)

        # 2. quality assessment
        qa = assess_image(img, cfg["quality"])
        qa["file"] = slide.name
        qa_rows.append(qa)
        if not qa["passed"]:
            logger.warning("  %s FLAGGED (%s) - excluded", slide.name, qa["flag_reason"])
            continue

        # 3. stain normalisation (+ before/after figure)
        norm = normalizer.normalize(img)
        save_rgb(Path(cfg["paths"]["interim_dir"]) / f"{slide_id}_norm.png", norm)
        save_before_after(
            img, norm,
            figures_dir / f"stain_normalization_{slide_id}.png",
            f"Macenko stain normalisation — {slide_id}",
        )

        # 4. patch extraction. Label comes from the slide's REAL source:
        #    a CATCH subtype parent folder when present (data/raw/<Subtype>/...),
        #    otherwise the demo filename convention. NEVER a colour-cast guess.
        label = slide_label(slide, subtypes)
        out_dir = patches_root / label
        n = extract_and_save(norm, out_dir, slide_id, cfg["patches"])
        logger.info("  %s -> %d patches (label=%s)", slide.name, n, label)

        # grid figure of a few patches
        sample = [p for (_, _, p) in extract_patches(
            norm, cfg["patches"]["patch_size"], cfg["patches"]["stride"],
            cfg["patches"]["min_tissue_fraction"], 16)]
        save_patch_grid(sample, figures_dir / f"patches_{slide_id}.png",
                        f"Sample patches — {slide_id} ({label})")

        # carry slide_id so the split groups by slide (no patch leakage)
        for patch_file in out_dir.glob(f"{slide_id}_*.png"):
            split_rows.append({"patch_path": str(patch_file), "label": label,
                               "slide_id": slide_id})

    # ---- QA report ----
    qa_df = pd.DataFrame(qa_rows)
    qa_csv = logs_dir / "quality_report.csv"
    qa_df.to_csv(qa_csv, index=False)
    logger.info("Quality report written: %s", qa_csv)

    # ---- 5. dataset split ----
    if split_rows:
        split_df = pd.DataFrame(split_rows)
        splits_dir = Path(cfg["paths"]["processed_dir"]) / "splits"
        summary = make_split(
            split_df, splits_dir,
            train=cfg["split"]["train"], val=cfg["split"]["val"],
            test=cfg["split"]["test"], stratify=cfg["split"]["stratify"],
            seed=cfg["project"]["seed"],
            group_col="slide_id",   # slide-level split -> no patch leakage
        )
        with open(logs_dir / "split_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Split: train=%d val=%d test=%d (total=%d)",
                    summary["train"], summary["val"], summary["test"], summary["total"])

    logger.info("DONE. Figures in %s, reports in %s", figures_dir, logs_dir)


if __name__ == "__main__":
    main()
