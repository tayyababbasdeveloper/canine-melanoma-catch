"""Week 3-4 segmentation pipeline — single entry point.

Runs the full Phase-2 (Model Development, start) workflow end-to-end and writes
presentation-ready artefacts for the supervisor meeting:

    1. prepare segmentation data (demo slides+masks -> Macenko -> multi-mag
       paired patches -> slide-level train/val/test manifests)
    2. train the BASELINE U-Net (from scratch)            -> baseline metrics
    3. train the ResNet-34-encoder U-Net (transfer learn) -> better metrics
    4. write a comparison report + figures

This mirrors the proposal Gantt: "U-Net segmentation, beginning with a baseline
architecture before incorporating the ResNet-34 encoder".

Usage:
    python scripts/run_week3_4_pipeline.py --demo --quick   # fast CPU check
    python scripts/run_week3_4_pipeline.py --demo           # full demo run
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


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Week 3-4 segmentation pipeline")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n-slides", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--baseline-epochs", type=int, default=None,
                    help="override epochs for the baseline only")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    prep = [PY, str(ROOT / "scripts" / "prepare_segmentation_data.py")]
    if args.demo:
        prep += ["--demo", "--n-slides", str(args.n_slides)]
    run(prep)

    common = []
    if args.quick:
        common += ["--quick"]
    if args.max_train:
        common += ["--max-train", str(args.max_train)]
    if args.max_eval:
        common += ["--max-eval", str(args.max_eval)]

    base_epochs = args.baseline_epochs or args.epochs
    base_extra = (["--epochs", str(base_epochs)] if base_epochs else [])
    res_extra = (["--epochs", str(args.epochs)] if args.epochs else [])

    run([PY, str(ROOT / "scripts" / "train_unet.py"),
         "--arch", "baseline", "--run-name", "unet_baseline"] + common + base_extra)
    run([PY, str(ROOT / "scripts" / "train_unet.py"),
         "--arch", "unet", "--run-name", "unet_resnet34"] + common + res_extra)

    # ---- comparison ----
    from src.utils.config import load_config
    cfg = load_config()
    logs = Path(cfg["paths"]["logs_dir"])
    rows = {}
    for run_name in ("unet_baseline", "unet_resnet34"):
        rp = logs / f"{run_name}_test_report.json"
        if rp.exists():
            rows[run_name] = json.loads(rp.read_text())

    comparison = {
        name: {
            "test_dice": r["test_metrics"]["dice"],
            "test_iou": r["test_metrics"]["iou"],
            "params_million": r["params_million"],
            "best_epoch": r["best_epoch"],
        } for name, r in rows.items()
    }
    out = logs / "week3_4_comparison.json"
    out.write_text(json.dumps(comparison, indent=2))

    print("\n================ Week 3-4 summary ================")
    for name, m in comparison.items():
        print(f"  {name:16s} dice={m['test_dice']:.4f} iou={m['test_iou']:.4f} "
              f"params={m['params_million']}M")
    print(f"Comparison written: {out}")
    print("TensorBoard:  tensorboard --logdir outputs/tensorboard")


if __name__ == "__main__":
    main()
