"""Train and evaluate the U-Net tumour segmentation model (Week 3-4).

Loads the segmentation manifests prepared by prepare_segmentation_data.py,
builds the model (baseline U-Net or ResNet-34-encoder U-Net), trains with the
proposal's recipe (Adam 1e-4 + cosine annealing + BCE/Dice, TensorBoard,
early stopping), then evaluates on the test split and writes figures.

Usage:
    python scripts/train_unet.py                       # config defaults (resnet34)
    python scripts/train_unet.py --arch baseline       # from-scratch baseline
    python scripts/train_unet.py --epochs 5 --quick    # fast end-to-end check
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger
from src.utils.seed import seed_everything
from src.data.segmentation_dataset import SegmentationPatchDataset
from src.data.augmentation import train_transform, eval_transform
from src.models.unet import build_model, count_parameters
from src.models.losses import build_loss
from src.training.trainer import SegTrainer, resolve_device
from src.training.evaluate import (evaluate, save_prediction_grid,
                                   save_training_curves)


def main():
    ap = argparse.ArgumentParser(description="Train U-Net segmentation")
    ap.add_argument("--arch", choices=["unet", "baseline"], default=None,
                    help="override segmentation.arch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="subsample data for a fast end-to-end check")
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap #train patches (stratified by tumour presence)")
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap #val and #test patches (stratified)")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    seed_everything(cfg["project"]["seed"])

    seg = cfg["segmentation"]
    if args.arch:        seg["arch"] = args.arch
    if args.epochs:      seg["epochs"] = args.epochs
    if args.batch_size:  seg["batch_size"] = args.batch_size
    run_name = args.run_name or f"unet_{seg['arch']}_{seg['encoder']}"

    logger = get_logger("train_unet", cfg["paths"]["logs_dir"])
    splits_dir = Path(seg["paths"]["seg_splits_dir"])
    for name in ("train", "val", "test"):
        if not (splits_dir / f"{name}.csv").exists():
            logger.error("Missing %s.csv — run prepare_segmentation_data.py first.",
                         name)
            return

    # ---- datasets / loaders ----
    import pandas as pd
    tr_df = pd.read_csv(splits_dir / "train.csv")
    va_df = pd.read_csv(splits_dir / "val.csv")
    te_df = pd.read_csv(splits_dir / "test.csv")

    def cap(df, n, seed=42):
        """Stratified subsample to <= n rows, preserving the positive/negative
        (tumour vs tumour-free) balance so capped sets stay representative."""
        if n is None or len(df) <= n:
            return df
        pos = df[df["tumour_frac"] >= 0.05]
        neg = df[df["tumour_frac"] < 0.05]
        frac = n / len(df)
        keep = pd.concat([
            pos.sample(max(1, int(round(len(pos) * frac))), random_state=seed),
            neg.sample(max(1, int(round(len(neg) * frac))), random_state=seed),
        ])
        return keep.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    max_train, max_eval = args.max_train, args.max_eval
    if args.quick:
        max_train = max_train or 64
        max_eval = max_eval or 32
        seg["epochs"] = min(seg["epochs"], 5)
    tr_df = cap(tr_df, max_train)
    va_df = cap(va_df, max_eval)
    te_df = cap(te_df, max_eval)

    ps = seg["patch_size"]
    train_ds = SegmentationPatchDataset(tr_df, transform=train_transform(cfg, ps))
    val_ds = SegmentationPatchDataset(va_df, transform=eval_transform(ps))
    test_ds = SegmentationPatchDataset(te_df, transform=eval_transform(ps))

    nw = int(seg.get("num_workers", 0))
    bs = int(seg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    logger.info("Patches: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    # ---- model / loss ----
    device = resolve_device(seg.get("device", "auto"))
    logger.info("Device: %s", device)
    model = build_model(cfg, logger=logger)
    logger.info("Model: %s (encoder=%s, params=%.2fM)",
                seg["arch"], seg["encoder"], count_parameters(model) / 1e6)

    pos_frac = train_ds.positive_fraction()
    pos_weight = (1 - pos_frac) / pos_frac if 0 < pos_frac < 1 else None
    logger.info("Tumour pixel fraction ~%.3f -> BCE pos_weight=%.2f",
                pos_frac, pos_weight or 0.0)
    loss_fn = build_loss(cfg, pos_weight=pos_weight)

    # ---- train ----
    trainer = SegTrainer(
        model, loss_fn, cfg, device, logger,
        ckpt_dir=Path(seg["paths"]["checkpoints_dir"]),
        tb_dir=Path(seg["paths"]["tensorboard_dir"]),
        run_name=run_name,
    )
    summary = trainer.fit(train_loader, val_loader)
    logger.info("Best val Dice %.4f at epoch %d",
                summary["best_val_dice"], summary["best_epoch"])

    # ---- evaluate on test ----
    import torch
    ckpt = torch.load(summary["best_checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device)
    logger.info("TEST | dice %.4f | iou %.4f | pix-acc %.4f | hausdorff %.2f",
                test_metrics["dice"], test_metrics["iou"],
                test_metrics["pixel_acc"], test_metrics["hausdorff"])

    # ---- figures + report ----
    figures_dir = Path(cfg["paths"]["figures_dir"])
    save_training_curves(summary["history"], figures_dir / f"{run_name}_curves.png",
                         title=f"{run_name} — training")
    save_prediction_grid(model, test_ds, device,
                         figures_dir / f"{run_name}_predictions.png",
                         title=f"{run_name} — test predictions")

    report = {
        "run_name": run_name, "arch": seg["arch"], "encoder": seg["encoder"],
        "device": str(device), "epochs_ran": summary["epochs_ran"],
        "best_epoch": summary["best_epoch"], "best_val_dice": summary["best_val_dice"],
        "test_metrics": test_metrics,
        "params_million": round(count_parameters(model) / 1e6, 2),
    }
    out = Path(cfg["paths"]["logs_dir"]) / f"{run_name}_test_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written: %s", out)


if __name__ == "__main__":
    main()
