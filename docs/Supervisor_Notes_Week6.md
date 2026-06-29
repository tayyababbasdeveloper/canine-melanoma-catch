# Supervisor Meeting Notes — Week 6

**Project:** Classification of Malignant Melanoma in Canines (CATCH) — DAIM2025A_088
**Student:** Muhammad Tayyab Abbas · **Supervisor:** Dr Claire Cashmore · University of Hull
**Meeting:** Third fortnightly supervisor meeting · **Date:** 29 June 2026

These are my talking points for the meeting — what I did, where it honestly stands,
the issues I found and fixed, and the decisions I need from you.

---

## 1. What I set out to do (Weeks 5–6)

- Add the **Attention U-Net** (Oktay et al., 2018) and produce a **3-way
  segmentation comparison** (baseline vs. ResNet-34 encoder vs. attention).
- Start the **classification track**: a **ResNet-50** tumour-subtype classifier with
  ImageNet transfer learning.
- Get the project **ready for the real CATCH data**, not just the demo.

All three are done.

## 2. Honest data status (please read first)

- **No real CATCH slides have been processed yet.** Everything below is validated on
  **synthetic** slides while the TCIA download is arranged. The synthetic "tumours"
  and "subtypes" are drawn shapes/colours — they prove the *code* works, **not** the
  histology.
- I now know the dataset precisely and have wired the code to it: **350 `.svs` WSIs,
  50 each of 7 subtypes, 12,424 polygon annotations** (MS-COCO + SQLite), TCIA DOI
  `10.7937/TCIA.2M93-FX66`.
- The blocker is the download itself: it is **license-gated and large (~hundreds of
  GB)**, and needs the **OpenSlide** binaries installed. I'd like your steer on
  scope (all 350 vs. a subset to start).

## 3. What I built

**Models / training**
- Attention gates on every skip connection, on the **same backbone** as the baseline
  U-Net (so the comparison isolates the attention effect — a clean controlled
  experiment).
- ResNet-50 classifier: progressive unfreezing (head → deeper layers), class-weighted
  cross-entropy + label smoothing, AdamW, early stopping on macro-F1.

**Real-data readiness (new this fortnight)**
- `src/preprocessing/wsi.py` — tiles real `.svs` at a chosen magnification via
  OpenSlide, tile-by-tile (never the whole slide in memory).
- `src/preprocessing/catch_annotations.py` — parses the CATCH COCO/SQLite polygons
  and rasterises a **binary tumour mask per tile**.
- The 7 official subtypes, folder-per-subtype labelling, and a real download helper
  (DOI/NBIA workflow) are all wired through config.

## 4. Three bugs I found and fixed (worth flagging)

1. **Label bug (Week 1–2):** subtype labels were derived from a colour-cast in the
   filename — biologically meaningless and wrong on real slides. Now taken from the
   real source (subtype folder / annotation).
2. **Data leakage:** the Week 1–2 patch split mixed patches from the *same slide*
   across train/test, which inflates metrics. The split is now **slide-level**
   (verified: zero slide overlap; correct 238/56/56 across the 350-slide set).
3. **Transfer-learning bug:** the classifier's "frozen" encoder BatchNorm was still
   updating its statistics during training, overwriting the pretrained ImageNet
   stats. Frozen BatchNorm is now held in eval mode (verified: 0 drift when frozen,
   updates when unfrozen).

## 5. Results (synthetic — validate the pipeline, not the biology)

**Segmentation (3-way):**

| Model | Params | Dice ↑ | IoU ↑ | Hausdorff ↓ |
|-------|-------:|-------:|------:|------------:|
| Baseline U-Net | 31.0 M | 0.218 | 0.204 | 154.8 |
| U-Net + ResNet-34 | 24.4 M | 0.430 | 0.405 | 115.1 |
| **Attention U-Net** | 31.4 M | **0.920** | **0.906** | **19.6** |

The *ranking* (attention > pretrained encoder > from-scratch) matches the
literature; the *absolute* numbers reflect easy synthetic masks + a short CPU run.

**Classification (ResNet-50):** accuracy / F1 / AUC = **1.000** — but this is an
**artefact of colour-separable synthetic classes**, not subtype discrimination. On
real CATCH it becomes a 7-class, class-imbalanced problem.

## 6. Decisions I need from you

1. **Primary classification target:** full **7-class** subtype, or **binary
   melanoma-vs-rest**? (Code supports both via `classification.mode`.)
2. **Combined vs. separate reporting:** one end-to-end (segmentation→classification)
   metric, or two separate evaluations?
3. **Data scope to start:** is a subset of subtypes acceptable for the first real run,
   or should I wait for the full 350?
4. **Interim evaluation:** single split now, with 5-fold CV reserved for the final?

## 7. Next fortnight (Weeks 7–8)

- EfficientNet-B3 classifier + comparison with ResNet-50.
- First real-data run once OpenSlide + the download are in place.
- Begin the evaluation phase: 5-fold cross-validation, McNemar's test, Grad-CAM.

---
*Artefacts: `outputs/logs/week5_6_seg_comparison.json`, `cls_resnet50_test_report.json`;
figures in `outputs/figures/`; full report in `docs/progress_report/Progress_Report_Week5-6.md`.*
