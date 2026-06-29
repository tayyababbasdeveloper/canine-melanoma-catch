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
from src.preprocessing.wsi import WSI_EXTENSIONS, open_slide, iter_tiles, tile_geometry
from src.preprocessing.catch_annotations import (
    find_annotation_file, load_annotations, rasterise_tile_mask,
    subtype_from_filename)
from src.preprocessing.tissue import tissue_mask


def _label_from_path(p: Path, subtypes: set[str], prefix_map: dict) -> str:
    """Subtype label for a slide, in priority order:
      1. a parent folder naming a CATCH subtype (data/raw/<Subtype>/slide.svs);
      2. the real CATCH filename prefix ('MCT_15_1.svs' -> 'Mast Cell Tumor');
      3. the demo filename convention '<label>_<n>'.
    """
    for parent in p.parents:
        if parent.name in subtypes:
            return parent.name
    st = subtype_from_filename(p.name, prefix_map)
    if st:
        return st
    return p.stem.rsplit("_", 1)[0]


def _list_labeled(input_dir: Path, subtypes: set[str], prefix_map: dict):
    """Label demo/real PNG slides. Looks in ``cls_images/`` first, then anywhere
    a subtype folder is found (real CATCH may put .svs directly under data/raw/)."""
    search = (input_dir / "cls_images") if (input_dir / "cls_images").exists() else input_dir
    pairs = []
    for p in sorted(search.rglob("*")):
        if p.suffix.lower() in STD_EXTENSIONS and p.parent.name != "masks":
            pairs.append((p, _label_from_path(p, subtypes, prefix_map)))
    return pairs


def _list_wsi_labeled(input_dir: Path, subtypes: set[str], prefix_map: dict):
    """Real CATCH .svs slides labelled by subtype (folder or filename prefix)."""
    return [(p, _label_from_path(p, subtypes, prefix_map))
            for p in sorted(input_dir.rglob("*"))
            if p.suffix.lower() in WSI_EXTENSIONS and "demo" not in p.stem
            and "annotations" not in p.parts]


def _process_real_wsi_classification(wsi_pairs, cfg, normalizer, patches_dir,
                                     label_idx, logger):
    """Extract TUMOUR-region tiles from real CATCH WSIs for subtype classification.

    Implements the project's two-stage plan on real data: the polygon annotations
    select the tumour tiles (the role the U-Net plays at inference time), and each
    tumour tile inherits its slide's subtype label. Returns manifest rows.
    """
    catch = cfg["catch"]
    c = cfg["classification"]
    ann_path = find_annotation_file(cfg["paths"]["raw_dir"],
                                    catch["coco_annotation_glob"]
                                    + catch["sqlite_annotation_glob"])
    per_slide = (load_annotations(ann_path, catch["tumour_annotation_classes"])
                 if ann_path else {})
    if ann_path:
        logger.info("Using annotations for tumour-tile selection: %s", ann_path)
    else:
        logger.warning("No annotations found — keeping all tissue tiles instead of "
                       "tumour-only tiles.")

    read_mag = catch["wsi_read_magnification"]
    base_mag = catch["wsi_base_magnification"]
    min_tf = cfg["patches"]["min_tissue_fraction"]
    max_p = cfg["patches"]["max_patches_per_slide"]
    tumour_tile_frac = 0.25  # tile counts as tumour if >=25% of pixels are tumour

    rows = []
    for sp, label in wsi_pairs:
        slide_id = sp.stem
        polys = per_slide.get(slide_id, [])
        slide = open_slide(sp)
        geom = tile_geometry(slide, c["patch_size"], read_mag, base_mag)
        span0 = geom["span0"]
        n = 0
        for (x0, y0, patch) in iter_tiles(slide, patch_size=c["patch_size"],
                                          read_magnification=read_mag,
                                          base_magnification=base_mag):
            if tissue_mask(patch).mean() < min_tf:
                continue
            if polys is not None and per_slide:  # tumour-only selection
                m = rasterise_tile_mask(polys, x0, y0, span0, c["patch_size"])
                if m.mean() < tumour_tile_frac:
                    continue
            try:
                patch_n = normalizer.normalize(patch)
            except Exception:
                patch_n = patch
            stem = f"{slide_id}_y{y0}_x{x0}"
            ip = patches_dir / label / f"{stem}.png"
            save_rgb(ip, patch_n)
            rows.append({
                "image_path": Path(ip).resolve().relative_to(PROJECT_ROOT).as_posix(),
                "label": label, "label_idx": label_idx[label], "slide_id": slide_id,
            })
            n += 1
            if n >= max_p:
                break
        logger.info("  %-22s [%s] -> %d tumour tiles", slide_id, label, n)
    return rows


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

    subtypes = set(cfg["catch"]["subtypes"])
    prefix_map = cfg["catch"].get("filename_prefix_map", {})
    normalizer = MacenkoNormalizer(**cfg["stain_normalization"])
    input_dir = Path(args.input) if args.input else Path(cfg["paths"]["raw_dir"])

    if args.demo:
        logger.info("DEMO mode: generating %d slides/class for %s",
                    args.slides_per_class, list(CLASSES))
        pairs = make_demo_classification_slides(
            cfg["paths"]["raw_dir"], slides_per_class=args.slides_per_class)
        wsi_pairs = []
    else:
        # REAL data: prefer whole-slide .svs (labelled by subtype folder); fall
        # back to pre-rendered PNGs.
        wsi_pairs = _list_wsi_labeled(input_dir, subtypes, prefix_map)
        pairs = [] if wsi_pairs else _list_labeled(input_dir, subtypes, prefix_map)

    if not pairs and not wsi_pairs:
        logger.error("No labelled slides found. Use --demo, or place real CATCH "
                     "slides under data/raw/<Subtype>/*.svs "
                     "(python -m src.data_acquisition.download_catch).")
        return

    classes = sorted({lbl for _, lbl in (pairs + wsi_pairs)})
    label_idx = {lbl: i for i, lbl in enumerate(classes)}
    logger.info("Classes (%s mode): %s", c.get("mode", "subtype"), label_idx)

    rows = []

    # ---- real WSIs: keep tumour-region tiles, label by subtype ----
    if wsi_pairs:
        rows = _process_real_wsi_classification(
            wsi_pairs, cfg, normalizer, patches_dir, label_idx, logger)

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

    if not rows:
        logger.error("No patches extracted — nothing to split.")
        return
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
