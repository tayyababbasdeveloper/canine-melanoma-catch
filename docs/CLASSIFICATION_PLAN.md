# Classification Section — Plan (answering the supervisor's Week 5/6 questions)

This document plans the classification stage and how it connects to the
segmentation output, directly addressing the supervisor's questions.

---

## The two-stage pipeline (segmentation → classification)

```
Whole-slide image (WSI)
  └─ tile into 256×256 patches at 5× / 10× / 20×
       └─ STAGE 1: U-Net SEGMENTATION  → per-pixel tumour MASK (tumour vs background)
            └─ select the tumour-containing patches  (tumour localisation)
                 └─ STAGE 2: CNN CLASSIFICATION (ResNet-50 / EfficientNet-B3)
                      → tumour SUBTYPE label
```

The segmentation **localises** the tumour; the classifier then **names the
subtype** of the tumour tissue. Each stage has its own model, loss and metrics.

---

## Q1 — What is the output of the segmentation?

A **per-pixel binary mask** for each patch: `1 = tumour`, `0 = background / normal
tissue`. Stitching the patch masks back together gives a **tumour-region map for
the whole slide**.

This output is used in two ways:
1. **Localisation** — where the tumour is (the deliverable of the segmentation task).
2. **Patch selection** — only the patches the mask marks as *tumour* are passed to
   the classifier, so the classifier sees tumour tissue, not empty background.

Metrics for this output: **Dice, IoU, pixel-accuracy, Hausdorff** (already
implemented and reported for baseline / ResNet-34 / Attention U-Net).

---

## Q2 — What are we going to classify, and how many classes?

We classify the **tumour subtype**. The real **CATCH** dataset has **7 tumour
subtypes**:

| # | Subtype |
|---|---------|
| 1 | Melanoma |
| 2 | Mast cell tumour |
| 3 | Squamous cell carcinoma |
| 4 | Peripheral nerve sheath tumour |
| 5 | Trichoblastoma |
| 6 | Histiocytoma |
| 7 | Plasmacytoma |

Because the project's focus is **melanoma**, the classifier supports two modes:
- **(A) 7-class** subtype classification (full multi-class), or
- **(B) Melanoma vs. non-melanoma** (binary) — sharper for the melanoma research question.

The current demo validates the pipeline on **3 representative classes**
(melanocytic, mast cell, squamous cell). Switching to 7 classes or binary is a
one-line change (the class list is read from the data, not hard-coded).

---

## Q3 — Are there enough of each class? (balance)

CATCH is **imbalanced** — melanocytic and mast-cell tumours are common, some other
subtypes are rarer. The pipeline **measures and reports** the per-class patch
counts every run (see `cls_split_summary.json` and the notebook bar chart), so the
imbalance is always explicit rather than hidden.

If a class is too small for reliable training, the options (in order) are:
group rare subtypes, switch to the **binary melanoma vs. non-melanoma** task, or
rely more heavily on the balancing measures below.

---

## Q4 — Balancing: before or after the segmentation?

There are **two different imbalances**, handled at **two different stages**:

| Imbalance | Where | How it is handled |
|-----------|-------|-------------------|
| Tumour pixels (~9 %) vs background | **inside segmentation** | BCE **`pos_weight`** + **Dice** loss |
| Subtype counts (class imbalance) | **after segmentation, in classification** | **class-weighted cross-entropy** + augmentation + **stratified slide-level split** |

So **class balancing for the subtypes is done AFTER segmentation**, on the tumour
patches that are fed to the classifier — because the class label only matters once
we have isolated tumour tissue. (A `WeightedRandomSampler` is an equivalent
alternative to the weighted loss.)

---

## What is already implemented (code)

| Component | File |
|-----------|------|
| ResNet-50 / EfficientNet-B3 classifier (transfer learning, progressive unfreezing) | `src/models/classifier.py` |
| Classification dataset (portable paths, slide-level split) | `src/data/classification_dataset.py`, `scripts/prepare_classification_data.py` |
| Trainer: class-weighted CE + label smoothing, AdamW, TensorBoard, early stopping | `src/training/classifier_trainer.py` |
| Metrics: accuracy, macro P/R/F1, AUC-ROC, confusion matrix | `src/training/classifier_trainer.py` |
| End-to-end run | `scripts/run_week5_6_pipeline.py`, `scripts/train_classifier.py` |

---

## Dataset scale — *"are you using the full dataset?"*

The numbers seen so far (e.g. **616**) are **extracted patches**, not slides — one
slide yields many 256×256 patches across 5×/10×/20×. Development is currently
**validated on synthetic slides** while the **full CATCH (750 WSIs)** download from
TCIA completes. **The same code runs unchanged on the full dataset** — only the
input folder changes:

```bash
python scripts/prepare_classification_data.py --input data/raw   # real labelled slides
python scripts/train_classifier.py --arch resnet50
```

See `notebooks/CATCH_Melanoma_Project.ipynb` (Section 7) for the full-dataset run
commands.
