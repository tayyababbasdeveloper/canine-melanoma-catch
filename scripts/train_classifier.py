"""Train and evaluate a tumour-subtype classifier (Week 6).

Loads the classification manifests, builds ResNet-50 or EfficientNet-B3 with
ImageNet transfer learning, trains with the proposal's recipe (AdamW, label
smoothing, progressive unfreezing, TensorBoard, early stopping), then evaluates
on the test split and writes a confusion-matrix figure + report.

Usage:
    python scripts/train_classifier.py                          # resnet50
    python scripts/train_classifier.py --arch efficientnet_b3
    python scripts/train_classifier.py --epochs 6 --quick
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger
from src.utils.seed import seed_everything
from src.data.classification_dataset import ClassificationPatchDataset
from src.data.augmentation import cls_train_transform, cls_eval_transform
from src.models.classifier import build_classifier, count_trainable
from src.training.trainer import resolve_device
from src.training.classifier_trainer import (ClassifierTrainer, evaluate,
                                             save_confusion_matrix)


def main():
    ap = argparse.ArgumentParser(description="Train tumour-subtype classifier")
    ap.add_argument("--arch", choices=["resnet50", "efficientnet_b3"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(); ensure_dirs(cfg)
    seed_everything(cfg["project"]["seed"])
    c = cfg["classification"]
    if args.arch:   c["arch"] = args.arch
    if args.epochs: c["epochs"] = args.epochs
    run_name = args.run_name or f"cls_{c['arch']}"

    logger = get_logger("train_cls", cfg["paths"]["logs_dir"])
    splits_dir = Path(c["paths"]["cls_splits_dir"])
    for name in ("train", "val", "test"):
        if not (splits_dir / f"{name}.csv").exists():
            logger.error("Missing %s.csv — run prepare_classification_data.py first.", name)
            return

    summary_json = json.loads((splits_dir / "cls_split_summary.json").read_text())
    class_names = [k for k, _ in sorted(summary_json["classes"].items(),
                                        key=lambda kv: kv[1])]
    logger.info("Classes: %s", class_names)

    tr_df = pd.read_csv(splits_dir / "train.csv")
    va_df = pd.read_csv(splits_dir / "val.csv")
    te_df = pd.read_csv(splits_dir / "test.csv")
    if args.quick:
        tr_df = tr_df.groupby("label").head(40).reset_index(drop=True)
        c["epochs"] = min(c["epochs"], 6)

    train_ds = ClassificationPatchDataset(tr_df, transform=cls_train_transform(cfg))
    val_ds = ClassificationPatchDataset(va_df, transform=cls_eval_transform())
    test_ds = ClassificationPatchDataset(te_df, transform=cls_eval_transform())
    logger.info("Patches: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    nw, bs = int(c.get("num_workers", 0)), int(c["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    device = resolve_device(c.get("device", "auto"))
    logger.info("Device: %s", device)
    model = build_classifier(c["arch"], num_classes=len(class_names),
                             pretrained=c.get("pretrained", True), logger=logger)
    logger.info("Model: %s (trainable head params=%d)", c["arch"], count_trainable(model))

    # class weights for imbalance handling (inverse frequency, normalised)
    counts = tr_df["label_idx"].value_counts().sort_index()
    freqs = counts.reindex(range(len(class_names)), fill_value=1).values
    inv = freqs.sum() / (len(class_names) * freqs)
    class_weights = (inv / inv.mean()).tolist()

    trainer = ClassifierTrainer(
        model, cfg, device, logger, class_names,
        ckpt_dir=Path(cfg["segmentation"]["paths"]["checkpoints_dir"]),
        tb_dir=Path(cfg["segmentation"]["paths"]["tensorboard_dir"]),
        run_name=run_name, class_weights=class_weights)
    fit_summary = trainer.fit(train_loader, val_loader)
    logger.info("Best val macro-F1 %.4f at epoch %d",
                fit_summary["best_val_f1"], fit_summary["best_epoch"])

    # ---- test ----
    ckpt = torch.load(fit_summary["best_checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics, cm, _, _ = evaluate(model, test_loader, device, len(class_names))
    logger.info("TEST | acc %.3f | macro-F1 %.3f | precision %.3f | recall %.3f | auc %.3f",
                metrics["accuracy"], metrics["f1"], metrics["precision"],
                metrics["recall"], metrics["auc_roc"])

    figures_dir = Path(cfg["paths"]["figures_dir"])
    save_confusion_matrix(cm, class_names, figures_dir / f"{run_name}_confusion.png",
                          title=f"{run_name} — test confusion matrix")

    report = {"run_name": run_name, "arch": c["arch"], "device": str(device),
              "classes": class_names, "epochs_ran": fit_summary["epochs_ran"],
              "best_epoch": fit_summary["best_epoch"], "test_metrics": metrics,
              "confusion_matrix": cm.tolist()}
    out = Path(cfg["paths"]["logs_dir"]) / f"{run_name}_test_report.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("Report written: %s", out)


if __name__ == "__main__":
    main()
