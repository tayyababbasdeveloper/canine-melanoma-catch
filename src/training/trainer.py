"""Segmentation training loop (proposal sections 3.1.2 & 3.1.4).

Implements exactly what the proposal specifies:
  * Adam optimiser, initial learning rate 1e-4,
  * cosine annealing learning-rate schedule,
  * composite BCE + Dice loss,
  * TensorBoard monitoring of loss and validation metrics (Dice/IoU/...),
  * early stopping on validation Dice with best-checkpoint saving,
  * automatic mixed precision when a CUDA GPU is available.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.models.metrics import SegMetrics


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class SegTrainer:
    def __init__(self, model, loss_fn, cfg, device, logger,
                 ckpt_dir: Path, tb_dir: Path, run_name: str = "unet"):
        s = cfg["segmentation"]
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.logger = logger
        self.run_name = run_name

        self.epochs = int(s.get("epochs", 40))
        self.patience = int(s.get("early_stopping_patience", 10))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(s.get("lr", 1e-4)),
            weight_decay=float(s.get("weight_decay", 1e-4)),
        )
        self.scheduler = None
        if s.get("scheduler", "cosine") == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs,
            )
        self.use_amp = bool(s.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(Path(tb_dir) / run_name))
        self.history: list[dict] = []

    # ---- one epoch ----
    def _train_epoch(self, loader) -> float:
        self.model.train()
        running = 0.0
        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running += loss.item() * images.size(0)
        return running / len(loader.dataset)

    @torch.no_grad()
    def _validate(self, loader) -> tuple[float, dict]:
        self.model.eval()
        running = 0.0
        metrics = SegMetrics()
        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)
            running += loss.item() * images.size(0)
            metrics.update(logits, masks)
        return running / len(loader.dataset), metrics.compute()

    # ---- full fit ----
    def fit(self, train_loader, val_loader) -> dict:
        best_dice, best_epoch, since_improve = -1.0, -1, 0
        best_path = self.ckpt_dir / f"{self.run_name}_best.pt"

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)
            if self.scheduler:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            # ---- TensorBoard ----
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("lr", lr, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            self.logger.info(
                "epoch %02d/%d | train %.4f | val %.4f | dice %.4f | iou %.4f | "
                "lr %.2e | %.1fs",
                epoch, self.epochs, train_loss, val_loss,
                val_metrics["dice"], val_metrics["iou"], lr, dt,
            )
            self.history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "lr": lr, **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            # ---- checkpoint + early stopping on val Dice ----
            if val_metrics["dice"] > best_dice:
                best_dice, best_epoch, since_improve = val_metrics["dice"], epoch, 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch,
                            "val_dice": best_dice}, best_path)
            else:
                since_improve += 1
                if since_improve >= self.patience:
                    self.logger.info(
                        "Early stopping at epoch %d (no val-Dice gain for %d epochs)",
                        epoch, self.patience)
                    break

        self.writer.close()
        summary = {
            "run_name": self.run_name,
            "best_epoch": best_epoch,
            "best_val_dice": best_dice,
            "epochs_ran": len(self.history),
            "best_checkpoint": str(best_path),
            "history": self.history,
        }
        with open(self.ckpt_dir / f"{self.run_name}_history.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary
