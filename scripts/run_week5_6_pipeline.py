"""Week 5-6 pipeline — single entry point.

Phase-2 continuation:
    Week 5-6  : Attention U-Net (segmentation) — compared against the Week 3-4
                baseline and ResNet-34 U-Nets (3-way comparison).
    Week 6    : ResNet-50 tumour-subtype classification (transfer learning).

Steps:
    1. train the Attention U-Net on the existing segmentation patches
    2. build a 3-way segmentation comparison (baseline / ResNet-34 / Attention)
    3. prepare the classification dataset (demo multi-class slides)
    4. train + evaluate the ResNet-50 classifier
    5. write a combined Week 5-6 summary

Usage:
    python scripts/run_week5_6_pipeline.py --demo --quick
    python scripts/run_week5_6_pipeline.py --demo
"""
from __future__ import annotations

import sys
import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable


def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Week 5-6 pipeline")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seg-epochs", type=int, default=None)
    ap.add_argument("--cls-epochs", type=int, default=None)
    ap.add_argument("--slides-per-class", type=int, default=4)
    args = ap.parse_args()

    from src.utils.config import load_config
    cfg = load_config()
    logs = Path(cfg["paths"]["logs_dir"])

    seg_common = (["--quick"] if args.quick else [])
    if args.seg_epochs:
        seg_common += ["--epochs", str(args.seg_epochs)]
    if not args.quick:
        seg_common += ["--max-train", "100", "--max-eval", "60"]

    # ---- 1. Attention U-Net (segmentation) ----
    if not (Path(cfg["segmentation"]["paths"]["seg_splits_dir"]) / "train.csv").exists():
        run([PY, ROOT / "scripts" / "prepare_segmentation_data.py", "--demo", "--n-slides", "14"])
    run([PY, ROOT / "scripts" / "train_unet.py",
         "--arch", "attention", "--run-name", "unet_attention"] + seg_common)

    # ---- 2. 3-way segmentation comparison ----
    seg_cmp = {}
    for name in ("unet_baseline", "unet_resnet34", "unet_attention"):
        rp = logs / f"{name}_test_report.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            seg_cmp[name] = {"test_dice": r["test_metrics"]["dice"],
                             "test_iou": r["test_metrics"]["iou"],
                             "params_million": r["params_million"]}
    (logs / "week5_6_seg_comparison.json").write_text(json.dumps(seg_cmp, indent=2))

    # ---- 3-4. ResNet-50 classification ----
    run([PY, ROOT / "scripts" / "prepare_classification_data.py",
         "--demo", "--slides-per-class", str(args.slides_per_class)])
    cls_common = (["--quick"] if args.quick else [])
    if args.cls_epochs:
        cls_common += ["--epochs", str(args.cls_epochs)]
    run([PY, ROOT / "scripts" / "train_classifier.py",
         "--arch", "resnet50", "--run-name", "cls_resnet50"] + cls_common)

    # ---- 5. combined summary ----
    cls_rep = logs / "cls_resnet50_test_report.json"
    cls_summary = {}
    if cls_rep.exists():
        r = json.loads(cls_rep.read_text())
        cls_summary = {"arch": r["arch"], **r["test_metrics"], "classes": r["classes"]}
    combined = {"segmentation_3way": seg_cmp, "classification_resnet50": cls_summary}
    (logs / "week5_6_summary.json").write_text(json.dumps(combined, indent=2))

    print("\n================ Week 5-6 summary ================")
    print("Segmentation (Dice):")
    for k, v in seg_cmp.items():
        print(f"  {k:16s} dice={v['test_dice']:.4f} iou={v['test_iou']:.4f}")
    if cls_summary:
        print(f"Classification (ResNet-50): acc={cls_summary.get('accuracy',0):.3f} "
              f"macro-F1={cls_summary.get('f1',0):.3f}")
    print("Saved: outputs/logs/week5_6_summary.json")


if __name__ == "__main__":
    main()
