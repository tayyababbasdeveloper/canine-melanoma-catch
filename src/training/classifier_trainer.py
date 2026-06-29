"""Classification training + evaluation (proposal sections 3.1.3 & 3.1.4).

Implements the proposal's recipe:
  * cross-entropy loss with label smoothing (0.1),
  * AdamW optimiser with weight decay,
  * progressive unfreezing (head -> deeper layers),
  * TensorBoard logging, early stopping on validation macro-F1, best checkpoint.

Evaluation reports accuracy, macro precision/recall/F1, and (when probabilities
allow) macro AUC-ROC, plus a confusion matrix.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)

from src.models.classifier import (set_trainable_stage, count_trainable,
                                    freeze_bn_running_stats)


def classification_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    """Compute accuracy, macro P/R/F1 and (if possible) macro AUC-ROC."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    auc = float("nan")
    try:
        if len(np.unique(y_true)) == num_classes:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        pass
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc_roc": auc}


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    """Return (metrics dict, confusion matrix, y_true, y_pred)."""
    model.eval()
    ys, preds, probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        preds.extend(prob.argmax(1).tolist())
        probs.extend(prob.tolist())
        ys.extend(labels.tolist())
    ys, preds, probs = np.array(ys), np.array(preds), np.array(probs)
    metrics = classification_metrics(ys, preds, probs, num_classes)
    cm = confusion_matrix(ys, preds, labels=list(range(num_classes)))
    return metrics, cm, ys, preds


class ClassifierTrainer:
    def __init__(self, model, cfg, device, logger, class_names,
                 ckpt_dir: Path, tb_dir: Path, run_name: str,
                 class_weights=None):
        c = cfg["classification"]
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.arch = c.get("arch", "resnet50")
        self.run_name = run_name

        self.epochs = int(c.get("epochs", 20))
        self.patience = int(c.get("early_stopping_patience", 8))
        self.unfreeze_at = list(c.get("unfreeze_epochs", [2, 4]))  # epoch -> next stage
        self.base_lr = float(c.get("lr", 3e-4))
        self.weight_decay = float(c.get("weight_decay", 1e-4))
        # Class-weighted cross-entropy handles class imbalance (proposal risk
        # table). Weights are inversely proportional to class frequency.
        w = None
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32, device=device)
            logger.info("Class weights (imbalance handling): %s",
                        [round(float(x), 2) for x in class_weights])
        self.loss_fn = nn.CrossEntropyLoss(
            weight=w, label_smoothing=float(c.get("label_smoothing", 0.1)))

        self.freeze_bn = bool(c.get("freeze_bn", True))
        self.stage = 0
        set_trainable_stage(self.model, self.arch, self.stage)
        self.optimizer = self._make_optimizer()
        self.use_amp = bool(c.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.ckpt_dir = Path(ckpt_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(Path(tb_dir) / run_name))
        self.history = []

    def _make_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.base_lr, weight_decay=self.weight_decay)

    def _maybe_unfreeze(self, epoch: int):
        # advance one unfreezing stage when we hit a scheduled epoch
        if self.stage < len(self.unfreeze_at) and epoch > self.unfreeze_at[self.stage]:
            self.stage += 1
            set_trainable_stage(self.model, self.arch, self.stage)
            self.optimizer = self._make_optimizer()
            self.logger.info("  unfroze to stage %d (%d trainable params)",
                             self.stage, count_trainable(self.model))

    def _train_epoch(self, loader) -> float:
        self.model.train()
        # keep frozen-encoder BatchNorm in eval mode so its pretrained running
        # stats are not overwritten by the new domain (transfer-learning fix)
        if self.freeze_bn:
            freeze_bn_running_stats(self.model)
        running = 0.0
        for images, labels in loader:
            images = images.to(self.device); labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss = self.loss_fn(self.model(images), labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer); self.scaler.update()
            running += loss.item() * images.size(0)
        return running / len(loader.dataset)

    def fit(self, train_loader, val_loader) -> dict:
        best_f1, best_epoch, since = -1.0, -1, 0
        best_path = self.ckpt_dir / f"{self.run_name}_best.pt"

        for epoch in range(1, self.epochs + 1):
            self._maybe_unfreeze(epoch)
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_metrics, _, _, _ = evaluate(self.model, val_loader,
                                            self.device, self.num_classes)
            dt = time.time() - t0

            self.writer.add_scalar("loss/train", train_loss, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.logger.info(
                "epoch %02d/%d | loss %.4f | val acc %.3f | val f1 %.3f | "
                "auc %.3f | %.1fs", epoch, self.epochs, train_loss,
                val_metrics["accuracy"], val_metrics["f1"],
                val_metrics["auc_roc"], dt)
            self.history.append({"epoch": epoch, "train_loss": train_loss,
                                 **{f"val_{k}": v for k, v in val_metrics.items()}})

            if val_metrics["f1"] > best_f1:
                best_f1, best_epoch, since = val_metrics["f1"], epoch, 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch,
                            "val_f1": best_f1, "classes": self.class_names}, best_path)
            else:
                since += 1
                if since >= self.patience:
                    self.logger.info("Early stopping at epoch %d", epoch)
                    break

        self.writer.close()
        summary = {"run_name": self.run_name, "best_epoch": best_epoch,
                   "best_val_f1": best_f1, "epochs_ran": len(self.history),
                   "best_checkpoint": str(best_path), "history": self.history}
        with open(self.ckpt_dir / f"{self.run_name}_history.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary


def save_confusion_matrix(cm, class_names, out_path: Path, title="Confusion matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)
