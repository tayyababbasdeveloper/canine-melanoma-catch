"""Prepare the segmentation patch dataset (Week 3).

Pipeline:
  1. (demo) generate synthetic H&E slides + tumour masks, or use real slides
     in data/raw/images + data/raw/masks.
  2. Macenko stain-normalise each slide.
  3. Multi-magnification (5x/10x/20x) paired image+mask patch extraction,
     keeping tissue-bearing tiles.
  4. Slide-level stratified split into train/val/test manifests (CSV) so that
     patches from one slide never span two subsets (prevents data leakage).

Usage:
    python scripts/prepare_segmentation_data.py --demo
    python scripts/prepare_segmentation_data.py --demo --n-slides 12
    python scripts/prepare_segmentation_data.py --input data/raw   # real slides
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger
from src.utils.seed import seed_everything
from src.utils.demo_segmentation import make_demo_segmentation_slides
from src.preprocessing.image_io import load_image_rgb
from src.preprocessing.stain_normalization import MacenkoNormalizer
from src.preprocessing.patch_extraction import (
    extract_multimag_with_mask, save_mask)
from src.preprocessing.image_io import save_rgb


def _list_pairs(input_dir: Path):
    """Pair data/raw/images/<id>.<ext> with data/raw/masks/<id>.<ext>."""
    img_dir, msk_dir = input_dir / "images", input_dir / "masks"
    pairs = []
    for ip in sorted(img_dir.rglob("*")):
        if ip.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        mp = msk_dir / ip.name
        if not mp.exists():  # try any extension
            cand = list(msk_dir.glob(ip.stem + ".*"))
            mp = cand[0] if cand else None
        if mp:
            pairs.append((ip, mp))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Prepare segmentation patches")
    ap.add_argument("--demo", action="store_true", help="generate synthetic slides")
    ap.add_argument("--n-slides", type=int, default=8, help="number of demo slides")
    ap.add_argument("--input", default=None, help="dir with images/ and masks/")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    seed_everything(cfg["project"]["seed"])
    logger = get_logger("prep_seg", cfg["paths"]["logs_dir"])

    seg = cfg["segmentation"]
    patches_dir = Path(seg["paths"]["seg_patches_dir"])
    splits_dir = Path(seg["paths"]["seg_splits_dir"])
    (patches_dir / "images").mkdir(parents=True, exist_ok=True)
    (patches_dir / "masks").mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. slide+mask pairs ----
    if args.demo:
        logger.info("DEMO mode: generating %d synthetic slides + masks", args.n_slides)
        pairs = make_demo_segmentation_slides(cfg["paths"]["raw_dir"], args.n_slides)
    else:
        input_dir = Path(args.input) if args.input else Path(cfg["paths"]["raw_dir"])
        pairs = _list_pairs(input_dir)
    if not pairs:
        logger.error("No (image, mask) pairs found. Use --demo or populate "
                     "data/raw/images and data/raw/masks.")
        return
    logger.info("Found %d slide/mask pair(s)", len(pairs))

    normalizer = MacenkoNormalizer(**cfg["stain_normalization"])
    mag = cfg["magnification"]
    rows = []

    # ---- 2-3. per slide: normalise + multi-mag patch/mask extraction ----
    for img_path, mask_path in pairs:
        slide_id = Path(img_path).stem
        original = load_image_rgb(img_path)
        mask = load_image_rgb(mask_path)[:, :, 0]  # masks are single channel

        # Stain-normalise the pixels we SAVE, but detect tissue on the ORIGINAL
        # slide (location of tissue must not depend on colour normalisation).
        try:
            img = normalizer.normalize(original)
        except Exception as exc:
            logger.warning("  %s: Macenko skipped (%s)", slide_id, exc)
            img = original

        n_slide = 0
        for (m, y, x, patch, mpatch) in extract_multimag_with_mask(
            img, mask, magnifications=mag["levels"], base_magnification=mag["base"],
            patch_size=seg["patch_size"], stride=seg["patch_size"],
            min_tissue_fraction=cfg["patches"]["min_tissue_fraction"],
            max_patches=cfg["patches"]["max_patches_per_slide"],
            tissue_img=original,
        ):
            stem = f"{slide_id}_m{m}_y{y}_x{x}"
            ip = patches_dir / "images" / f"{stem}.png"
            mp = patches_dir / "masks" / f"{stem}.png"
            save_rgb(ip, patch)
            save_mask(mp, mpatch)
            rows.append({
                "image_path": str(ip), "mask_path": str(mp),
                "slide_id": slide_id, "mag": m,
                "tumour_frac": round(float((mpatch > 127).mean()
                                           if mpatch.max() > 1 else mpatch.mean()), 4),
            })
            n_slide += 1
        logger.info("  %-14s -> %4d patches (mags=%s)", slide_id, n_slide, mag["levels"])

    df = pd.DataFrame(rows)
    df.to_csv(patches_dir / "manifest.csv", index=False)
    logger.info("Total patches: %d", len(df))

    # ---- 4. slide-level split (no slide spans two subsets) ----
    rng = np.random.default_rng(cfg["project"]["seed"])
    slides = sorted(df["slide_id"].unique())
    rng.shuffle(slides)
    n = len(slides)
    n_train = max(1, int(round(cfg["split"]["train"] * n)))
    n_val = max(1, int(round(cfg["split"]["val"] * n))) if n - n_train > 1 else 0
    train_s = set(slides[:n_train])
    val_s = set(slides[n_train:n_train + n_val])
    test_s = set(slides[n_train + n_val:])
    # guarantee non-empty test when possible
    if not test_s and len(slides) >= 3:
        test_s = {slides[-1]}; val_s.discard(slides[-1]); train_s.discard(slides[-1])

    def subset(name, sset):
        sub = df[df["slide_id"].isin(sset)]
        sub.to_csv(splits_dir / f"{name}.csv", index=False)
        return sub

    tr, va, te = subset("train", train_s), subset("val", val_s), subset("test", test_s)
    summary = {
        "slides": {"train": sorted(train_s), "val": sorted(val_s), "test": sorted(test_s)},
        "patches": {"train": len(tr), "val": len(va), "test": len(te), "total": len(df)},
        "magnifications": mag["levels"],
        "mean_tumour_frac": round(float(df["tumour_frac"].mean()), 4),
    }
    with open(splits_dir / "seg_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Split (patches): train=%d val=%d test=%d | slides %d/%d/%d",
                len(tr), len(va), len(te), len(train_s), len(val_s), len(test_s))
    logger.info("DONE. Manifests in %s", splits_dir)


if __name__ == "__main__":
    main()
