"""Prepare the tumour-subtype classification dataset (Week 6).

Pipeline:
  1. (demo) generate synthetic multi-class slides, or use real slides in
     data/raw/cls_images/<label>_*.png.
  2. Macenko stain-normalise each slide.
  3. Extract tissue patches (image only) and label each by its slide's subtype.
  4. Slide-level stratified split into train/val/test manifests, so patches from
     one slide never span two subsets.

Usage:
    python scripts/prepare_classification_data.py --demo
    python scripts/prepare_classification_data.py --demo --slides-per-class 5
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.config import load_config, ensure_dirs, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.seed import seed_everything
from src.utils.demo_classification import make_demo_classification_slides, CLASSES
from src.preprocessing.image_io import load_image_rgb, save_rgb, STD_EXTENSIONS
from src.preprocessing.stain_normalization import MacenkoNormalizer
from src.preprocessing.patch_extraction import extract_patches


def _list_labeled(input_dir: Path):
    """Pair data/raw/cls_images/<label>_<n>.png with its label (prefix)."""
    img_dir = input_dir / "cls_images"
    pairs = []
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in STD_EXTENSIONS:
            label = p.stem.rsplit("_", 1)[0]
            pairs.append((p, label))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Prepare classification patches")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--slides-per-class", type=int, default=4)
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    cfg = load_config(); ensure_dirs(cfg)
    seed_everything(cfg["project"]["seed"])
    logger = get_logger("prep_cls", cfg["paths"]["logs_dir"])

    c = cfg["classification"]
    patches_dir = Path(c["paths"]["cls_patches_dir"])
    splits_dir = Path(c["paths"]["cls_splits_dir"])
    patches_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        logger.info("DEMO mode: generating %d slides/class for %s",
                    args.slides_per_class, list(CLASSES))
        pairs = make_demo_classification_slides(
            cfg["paths"]["raw_dir"], slides_per_class=args.slides_per_class)
    else:
        input_dir = Path(args.input) if args.input else Path(cfg["paths"]["raw_dir"])
        pairs = _list_labeled(input_dir)
    if not pairs:
        logger.error("No labelled slides found. Use --demo or populate "
                     "data/raw/cls_images/<label>_*.png")
        return

    classes = sorted({lbl for _, lbl in pairs})
    label_idx = {lbl: i for i, lbl in enumerate(classes)}
    logger.info("Classes: %s", label_idx)

    normalizer = MacenkoNormalizer(**cfg["stain_normalization"])
    rows = []
    for img_path, label in pairs:
        slide_id = Path(img_path).stem
        original = load_image_rgb(img_path)
        try:
            img = normalizer.normalize(original)
        except Exception as exc:
            logger.warning("  %s: Macenko skipped (%s)", slide_id, exc); img = original

        # Save normalised pixels, but detect tissue on the ORIGINAL slide (robust
        # even on heavily pigmented melanocytic slides that Macenko distorts).
        n = 0
        for (y, x, patch) in extract_patches(
            img, patch_size=c["patch_size"], stride=c["patch_size"],
            min_tissue_fraction=cfg["patches"]["min_tissue_fraction"],
            max_patches=cfg["patches"]["max_patches_per_slide"],
            tissue_img=original,
        ):
            stem = f"{slide_id}_y{y}_x{x}"
            ip = patches_dir / label / f"{stem}.png"
            save_rgb(ip, patch)
            rows.append({
                "image_path": Path(ip).resolve().relative_to(PROJECT_ROOT).as_posix(),
                "label": label, "label_idx": label_idx[label], "slide_id": slide_id,
            })
            n += 1
        logger.info("  %-18s [%s] -> %d patches", slide_id, label, n)

    df = pd.DataFrame(rows)
    df.to_csv(patches_dir / "manifest.csv", index=False)
    logger.info("Total patches: %d", len(df))

    # slide-level stratified split (group by slide, stratify by class)
    rng = np.random.default_rng(cfg["project"]["seed"])
    train_s, val_s, test_s = set(), set(), set()
    for label in classes:
        slides = sorted(df[df["label"] == label]["slide_id"].unique())
        rng.shuffle(slides)
        n = len(slides)
        # reserve one val and one test slide per class whenever possible
        if n >= 4:
            n_va, n_te = 1, 1
        elif n == 3:
            n_va, n_te = 1, 1
        elif n == 2:
            n_va, n_te = 0, 1
        else:
            n_va, n_te = 0, 0
        n_tr = n - n_va - n_te
        train_s |= set(slides[:n_tr])
        val_s |= set(slides[n_tr:n_tr + n_va])
        test_s |= set(slides[n_tr + n_va:])

    def subset(name, sset):
        sub = df[df["slide_id"].isin(sset)]
        sub.to_csv(splits_dir / f"{name}.csv", index=False)
        return sub

    tr, va, te = subset("train", train_s), subset("val", val_s), subset("test", test_s)
    summary = {"classes": label_idx,
               "patches": {"train": len(tr), "val": len(va), "test": len(te), "total": len(df)},
               "class_distribution": df["label"].value_counts().to_dict()}
    with open(splits_dir / "cls_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Split (patches): train=%d val=%d test=%d", len(tr), len(va), len(te))
    logger.info("DONE. Manifests in %s", splits_dir)


if __name__ == "__main__":
    main()
