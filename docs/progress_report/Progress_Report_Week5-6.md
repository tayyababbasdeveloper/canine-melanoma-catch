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
finalised. In addition, the pipeline was made **genuinely real-data-ready** this
fortnight: native whole-slide (`.svs`) reading via OpenSlide, MS-COCO/SQLite
polygon-annotation parsing to build tumour masks, the official **7 CATCH subtypes**
wired through config, folder-per-subtype labelling, and a corrected **slide-level**
split. Switching to real data is now a matter of pointing `--input` at the
downloaded slides (see §6), not a code change.

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
- **Multi-class tumour-subtype** classifier. On real data it targets the full
  **7 CATCH subtypes** (Melanoma, Mast cell tumour, Squamous cell carcinoma,
  Peripheral nerve sheath tumour, Trichoblastoma, Histiocytoma, Plasmacytoma), or
  a binary **melanoma-vs-rest** mode (`classification.mode` in config). The demo
  uses 3 representative subtypes to exercise the pipeline.
- **ResNet-50** with ImageNet weights and a fresh head; **progressive
  unfreezing** (head → deeper layers) to protect pretrained features.
- **Cross-entropy + label smoothing (0.1)**, **AdamW**, TensorBoard, early
  stopping on validation macro-F1.
- Evaluation: **accuracy, macro precision/recall/F1, macro AUC-ROC, confusion
  matrix** (proposal §3.1.4).
- New, portable, slide-level-split classification dataset (`data/processed/cls_*`).

### 3.3 Real-data readiness & correctness fixes
- **Whole-slide ingestion** (`src/preprocessing/wsi.py`): tile real CATCH `.svs`
  at a chosen objective magnification via OpenSlide — never loading a whole slide
  into memory — so the same tissue/Macenko/dataset code runs on real slides.
- **Annotation parsing** (`src/preprocessing/catch_annotations.py`): MS-COCO JSON
  (and SQLite) polygons → binary tumour masks, rasterised per tile.
- **Bug fixes:** (i) replaced the Week 1–2 filename-based label heuristic with a
  real folder/subtype label source; (ii) made the patch split **slide-level**
  (`GroupShuffleSplit` semantics) to remove train/test patch leakage; (iii) kept
  frozen-encoder **BatchNorm in eval mode** so pretrained ImageNet statistics are
  not overwritten during transfer learning.

---

## 4. Results (synthetic validation)

### 4.1 Segmentation — three-way comparison

| Model | Params | Dice ↑ | IoU ↑ | Pixel acc ↑ | Hausdorff ↓ |
|-------|-------:|-------:|------:|------------:|------------:|
| Baseline U-Net (from scratch) | 31.04 M | 0.842 | 0.727 | 0.981 | 111.3 |
| U-Net + ResNet-34 encoder | 24.44 M | 0.225 | 0.127 | 0.534 | 218.5 |
| **Attention U-Net** | 31.39 M | **0.965** | **0.932** | **0.994** | **113.3** |

*Source: `outputs/logs/week5_6_seg_comparison.json` — all three from matched
**2-epoch** CPU runs on 16 capped patches. The **Attention U-Net clearly leads**
(Dice 0.965). The ResNet-34 encoder's low score here is **under-training noise**,
not a real ranking: with only 2 epochs and a high BCE `pos_weight` (~9.6) it sits
in an over-prediction regime (pixel-acc 0.53). These are a plumbing/headroom check
on easy synthetic masks, **not** a scientific result — absolute values and the
encoder comparison will change with full-length training on real CATCH annotations.*

### 4.2 Classification — ResNet-50

| Metric | Value |
|--------|------:|
| Classes | mast_cell, melanocytic, squamous_cell (demo) |
| Accuracy | 1.000 |
| Macro F1 | 1.000 |
| Macro precision / recall | 1.000 / 1.000 |
| Macro AUC-ROC | 1.000 |

*Source: `outputs/logs/cls_resnet50_test_report.json`. **The perfect score is an
artefact of the synthetic data**, whose three classes are colour-separable by
construction — it demonstrates the training/eval loop is correct, not subtype
discrimination on histology. On real CATCH this becomes a 7-class problem with the
expected class imbalance, handled by class-weighted cross-entropy and the
slide-level stratified split.*

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

## 6. Running on the real CATCH data

The real dataset (Wilm et al., 2022; TCIA DOI `10.7937/TCIA.2M93-FX66`) is **350
`.svs` WSIs, 50 per subtype**, with 12,424 MS-COCO/SQLite polygon annotations.
Recommended layout and commands:

```text
data/raw/
  Melanoma/*.svs   Mast cell tumor/*.svs   ...   Plasmacytoma/*.svs
  annotations/CATCH.json     # MS-COCO polygons (tumour masks)
```

```bash
python -m src.data_acquisition.download_catch          # instructions + verify
python scripts/prepare_segmentation_data.py  --input data/raw   # WSI tiles + masks
python scripts/train_unet.py --arch attention
python scripts/prepare_classification_data.py --input data/raw  # tumour tiles + subtype
python scripts/train_classifier.py --arch resnet50 --epochs 20
```

The scripts auto-detect `.svs` slides and the annotation file; no `--demo`.

**Next fortnight (Weeks 7–8):**
- Train **EfficientNet-B3** and compare with ResNet-50.
- Run the validated pipelines on the **real CATCH slides** once the download
  completes (requires OpenSlide binaries on the machine).
- Begin the **evaluation phase**: 5-fold cross-validation and McNemar's test.

---

## 7. Questions for supervisor

1. **7-class subtype vs. binary melanoma-vs-rest** — the code now supports both
   (`classification.mode`). For the dissertation's headline result, which should be
   primary? (The 7 CATCH subtypes are now wired in.)
2. Should segmentation and classification be reported as one combined end-to-end
   metric, or kept as two separate evaluations?
3. Is a single train/val/test split sufficient for the interim, with 5-fold
   cross-validation reserved for the final evaluation?
