# Progress Report — Weeks 5–6 (Attention U-Net + ResNet-50 Classification)

**Project:** Classification of Malignant Melanoma in Canines (CATCH dataset)
**Project Code:** DAIM2025A_088
**Student:** Muhammad Tayyab Abbas
**Supervisor:** Dr Claire Cashmore
**Reporting period:** Weeks 5–6 (23 June – 6 July 2026)
**Meeting:** Third fortnightly supervisor meeting

---

## 1. Summary

Phase 2 continued on schedule. Two things were delivered this fortnight:

1. **Attention U-Net** (Oktay et al., 2018) was implemented and trained, giving a
   **three-way segmentation comparison** — baseline U-Net, ResNet-34 U-Net, and
   Attention U-Net.
2. The **classification track began**: a **ResNet-50** tumour-subtype classifier
   (ImageNet transfer learning, progressive unfreezing, label smoothing) was
   trained and evaluated with accuracy, macro-F1, AUC-ROC and a confusion matrix.

Everything was validated end-to-end on synthetic data while the CATCH download is
finalised; the code runs unchanged on the real slides.

---

## 2. Planned vs. actual (Gantt)

| Gantt task | Planned | Status |
|------------|---------|--------|
| Attention U-Net Implementation | W5–6 | ✅ Complete — 3-way comparison |
| ResNet-50 Classification | W6–7 | 🔄 Started (W6) — trained & evaluated |

---

## 3. What was completed

### 3.1 Attention U-Net (segmentation)
- Implemented **attention gates** on every skip connection: each gate uses the
  decoder (gating) signal to compute a soft attention map and re-weights the
  encoder skip features, suppressing background and focusing on tumour regions.
- Built on the **same backbone as the baseline U-Net**, so comparing the two
  isolates the contribution of attention (the proposal's controlled experiment).
- Trained with the same recipe as Week 3–4 (BCE+Dice, Adam 1e-4, cosine
  annealing, early stopping, TensorBoard).

### 3.2 ResNet-50 classification (new track)
- **Multi-class tumour-subtype** classifier over three demo subtypes
  (melanocytic, mast cell, squamous cell).
- **ResNet-50** with ImageNet weights and a fresh head; **progressive
  unfreezing** (head → deeper layers) to protect pretrained features.
- **Cross-entropy + label smoothing (0.1)**, **AdamW**, TensorBoard, early
  stopping on validation macro-F1.
- Evaluation: **accuracy, macro precision/recall/F1, macro AUC-ROC, confusion
  matrix** (proposal §3.1.4).
- New, portable, slide-level-split classification dataset (`data/processed/cls_*`).

---

## 4. Results (synthetic validation)

### 4.1 Segmentation — three-way comparison
<!-- SEG_TABLE -->
*(filled from `outputs/logs/week5_6_seg_comparison.json`)*

### 4.2 Classification — ResNet-50
<!-- CLS_TABLE -->
*(filled from `outputs/logs/cls_resnet50_test_report.json`)*

> These are synthetic-data results that validate the pipelines end-to-end;
> absolute values will change on the real CATCH slides. The proposal targets are
> Dice > 0.85 (segmentation) and 85–92 % accuracy (classification).

---

## 5. Artefacts produced

| Artefact | Location |
|----------|----------|
| Attention U-Net checkpoint + history | `outputs/checkpoints/unet_attention_*` |
| 3-way segmentation comparison | `outputs/logs/week5_6_seg_comparison.json` |
| ResNet-50 checkpoint + report | `outputs/checkpoints/cls_resnet50_*`, `outputs/logs/cls_resnet50_test_report.json` |
| Confusion matrix + prediction figures | `outputs/figures/cls_resnet50_confusion.png`, `unet_attention_predictions.png` |
| Classification dataset | `data/processed/cls_patches/`, `cls_splits/` |
| Source: attention U-Net, classifier | `src/models/attention_unet.py`, `src/models/classifier.py` |

---

## 6. Plan for next fortnight (Weeks 7–8)

- Train **EfficientNet-B3** classifier and compare with ResNet-50.
- Run the validated pipelines on the **real CATCH slides** once downloaded.
- Begin the **evaluation phase**: 5-fold cross-validation and McNemar's test to
  compare architectures.

---

## 7. Questions for supervisor

1. For the real data, which tumour subtypes should be the classification classes?
2. Should the segmentation and classification be combined into one report metric,
   or kept separate?
3. Is a single train/val/test split sufficient for the interim, with 5-fold
   cross-validation reserved for the final evaluation?
