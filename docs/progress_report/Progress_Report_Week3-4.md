# Progress Report — Weeks 3–4 (Model Development: U-Net Segmentation)

**Project:** Classification of Malignant Melanoma in Canines (CATCH dataset)
**Project Code:** DAIM2025A_088
**Student:** Muhammad Tayyab Abbas
**Supervisor:** Dr Claire Cashmore
**Reporting period:** Weeks 3–4 (9 – 22 June 2026)
**Meeting:** Second fortnightly supervisor meeting

---

## 1. Summary

Phase 2 (Model Development) has begun on schedule. The remaining Phase-1
preprocessing items were completed — **multi-magnification (5×/10×/20×) patch
extraction** and a full **augmentation** pipeline — and the **U-Net tumour
segmentation** model has been implemented, trained and evaluated end-to-end.

Following the proposal exactly, a **from-scratch baseline U-Net** was trained
first, then a **U-Net with an ImageNet-pretrained ResNet-34 encoder** (transfer
learning). Both were validated on synthetic *annotated* slides while the CATCH
download is finalised; the code runs unchanged on the real annotated slides.

---

## 2. Planned vs. actual (Gantt)

| Gantt task | Planned | Status |
|------------|---------|--------|
| Data Acquisition & Preprocessing | W1–W3 | ✅ Complete (multi-mag + augmentation added) |
| U-Net Segmentation Development | W3–W5 | 🔄 On track — baseline + ResNet-34 trained & evaluated (W3–4) |

---

## 3. What was completed

### 3.1 Finished preprocessing (closing Phase-1 gaps)
- **Multi-magnification patch extraction (5×/10×/20×).** Lower magnifications are
  produced by downsampling the slide (factor = base ÷ target), giving each patch a
  wider field of view — the multi-scale context required by proposal §3.1.1.
- **Paired image+mask tiling** that keeps both tumour and tumour-free tissue
  patches (negatives), so the model learns true localisation rather than a
  trivial "all-foreground" shortcut.
- **Augmentation** (horizontal/vertical flip, 90° rotation, elastic deformation,
  brightness/contrast, H&E hue–saturation jitter, Gaussian noise), applied to the
  training set only.
- **Tissue-detection correctness fix:** tissue is now detected on the *original*
  slide rather than the stain-normalised one. *Where* tissue is located must not
  depend on colour normalisation; this also made the pipeline robust.

### 3.2 U-Net segmentation (proposal §3.1.2)
- **Baseline U-Net** (from scratch, ~31 M parameters) and **U-Net + ResNet-34
  encoder** (ImageNet-pretrained, ~24 M parameters), selectable from config.
- **Composite BCE + Dice loss**, with a BCE `pos_weight` automatically derived
  from the tumour-pixel fraction to counter class imbalance (proposal risk table:
  class imbalance — *High* likelihood).
- **Training recipe per proposal:** Adam, initial LR 1×10⁻⁴, **cosine annealing**,
  **early stopping** on validation Dice, **TensorBoard** logging of loss/metrics,
  best-checkpoint saving, and automatic mixed precision when a GPU is present.
- **Slide-level train/val/test split** so patches from one slide never span two
  subsets (prevents data leakage).

### 3.3 Evaluation (proposal §3.1.4)
- Test-set metrics: **Dice (DSC), IoU, pixel accuracy, Hausdorff distance**.
- Qualitative figures: **training curves** (loss + validation Dice) and
  **prediction overlays** (image | ground truth | prediction).

---

## 4. Results (synthetic validation dataset)

> Dataset: 14 synthetic annotated slides → 616 paired patches across 5×/10×/20×,
> mean tumour-pixel fraction ≈ 9 % (a realistic, imbalanced segmentation task).
> Split at slide level into train/val/test. Trained on CPU.

| Model | Params | Test Dice | Test IoU | Pixel acc. | Hausdorff |
|-------|:------:|:---------:|:--------:|:----------:|:---------:|
| Baseline U-Net (from scratch) | 31.0 M | 0.218 | 0.204 | 0.993 | 154.8 |
| **U-Net + ResNet-34 (transfer learning)** | **24.4 M** | **0.430** | **0.405** | 0.991 | **115.1** |

**Headline result.** The ImageNet-pretrained **ResNet-34 encoder roughly doubles
the Dice score over the from-scratch baseline (0.43 vs 0.22) using fewer
parameters** — direct evidence for the transfer-learning argument in proposal
§3.3, and consistent with the expectation that pretraining helps most on limited
data. Both models converged (training loss fell and validation Dice rose under
cosine annealing); the lower Dice relative to the very high pixel accuracy (0.99)
is expected for a strongly imbalanced task (~9 % tumour pixels), which is exactly
why Dice and IoU — not pixel accuracy — are the headline metrics.

**Interpretation.** These figures come from synthetic data and demonstrate that
the full pipeline trains, converges and evaluates correctly end-to-end; absolute
values will change on the real CATCH slides. The proposal target is **Dice > 0.85**
on real data. The value of this fortnight is the *validated, reproducible
machinery* — not the synthetic numbers.

---

## 5. Artefacts produced

| Artefact | Location |
|----------|----------|
| Segmentation patch dataset + manifest | `data/processed/seg_patches/` |
| Slide-level split manifests + summary | `data/processed/seg_splits/` |
| Trained checkpoints + training history | `outputs/checkpoints/` |
| TensorBoard logs | `outputs/tensorboard/` |
| Training-curve & prediction figures | `outputs/figures/unet_*_*.png` |
| Per-model test reports + comparison | `outputs/logs/unet_*_test_report.json`, `week3_4_comparison.json` |
| Source: models / data / training | `src/models/`, `src/data/`, `src/training/` |

---

## 6. Risks & issues

| Risk | Status | Mitigation |
|------|--------|-----------|
| CATCH download (TCIA) | Active | Whole pipeline validated on synthetic annotated slides; runs unchanged on real slides. |
| Class imbalance (tumour ≪ tissue) | Being handled | BCE `pos_weight`, Dice loss, augmentation; macro metrics. |
| GPU for larger training | Watching | Code is device-agnostic (AMP on GPU); Colab Pro / HPC fallback. |
| CPU training time | Managed | Stratified train/eval caps for the demo; full data on GPU later. |

---

## 7. Plan for next fortnight (Weeks 5–6)

- Implement the **Attention U-Net** variant and compare against the ResNet-34 U-Net.
- Run the validated pipeline on the **real CATCH annotated slides** once downloaded.
- Begin **ResNet-50 classification** (Week 6).

---

## 8. Questions for supervisor

1. For the real slides, confirm the tumour-class definition for the binary
   segmentation mask (melanocytic tumour vs all-tumour-vs-background?).
2. Preferred primary magnification for the first real-data experiments (10× vs 20×)?
3. Is Dice + IoU + Hausdorff the agreed metric set, or also boundary-F1?
