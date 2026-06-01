# Progress Report — Weeks 1–2 (Foundation Phase)

**Project:** Classification of Malignant Melanoma in Canines (CATCH dataset)
**Project Code:** DAIM2025A_088
**Student:** Muhammad Tayyab Abbas
**Supervisor:** Dr Claire Cashmore
**Reporting period:** Week 1–2 (26 May – 8 June 2026)
**Meeting:** First fortnightly supervisor meeting

---

## 1. Summary

The Foundation phase is on schedule. The computational environment is fully
configured, the literature review has been refined, and the data acquisition and
preprocessing pipeline (Macenko stain normalisation → patch extraction →
stratified split) has been implemented and validated end-to-end. The pipeline was
verified on synthetic H&E slides while the CATCH dataset download from TCIA is
being finalised; it is ready to run unchanged on the real slides.

---

## 2. Planned vs. actual (Gantt chart)

| Gantt task | Planned weeks | Status | Evidence |
|------------|---------------|--------|----------|
| Environment Setup & Framework | W1 | ✅ Complete | `.venv`, `requirements.txt`, `requirements-lock.txt`, `setup_env.*` |
| Literature Review & Refinement | W1–W2 | ✅ Complete | `docs/literature/literature_review_notes.md` |
| Data Acquisition & Preprocessing | W1–W3 | 🔄 ~60% | code + QA/split reports + figures |

---

## 3. What was completed

### 3.1 Environment setup (Week 1)
- Python 3.12 in an **isolated virtual environment (`.venv`)** so project packages
  never affect other projects or the global interpreter.
- Dependencies: OpenCV, scikit-image, scikit-learn, OpenSlide, NumPy, pandas,
  matplotlib (PyTorch + Albumentations added for Week 3 model development).
- Reproducible setup: `requirements.txt`, `environment.yml`, pinned
  `requirements-lock.txt`, and one-command installers (`scripts/setup_env.ps1`/`.sh`).
- Git repository initialised with a clean, modular `src/` package structure.

### 3.2 Literature review refinement (Weeks 1–2)
- Consolidated the key references underpinning the methodology
  (U-Net, Attention U-Net, ResNet, EfficientNet, Macenko, Grad-CAM,
  canine comparative-oncology papers).
- Notes and method/finding summaries recorded in
  `docs/literature/literature_review_notes.md`.

### 3.3 Data acquisition and preprocessing (Weeks 1–3, in progress)
Implemented and validated the complete preprocessing chain:

1. **Acquisition helper** (`download_catch.py`) — documents the TCIA / NBIA Data
   Retriever workflow and verifies downloaded slides (count, integrity).
2. **Quality assessment** (`quality_check.py`) — per-slide checks for size, blur
   (variance of Laplacian), brightness and tissue fraction; flags and excludes
   poor slides with a documented reason. Output: `outputs/logs/quality_report.csv`.
3. **Macenko stain normalisation** (`stain_normalization.py`) — maps slides onto a
   common H&E colour appearance to remove scanner/lab colour variation.
   Output: before/after figures in `outputs/figures/`.
4. **Patch extraction** (`patch_extraction.py`) — tiles slides into 256×256 patches
   and keeps only tissue-bearing tiles (≥50% tissue).
5. **Stratified split** (`dataset_split.py`) — 70/15/15 train/val/test split
   preserving class distribution. Output: `data/processed/splits/*.csv`.

**Validation run (synthetic slides):** 2 slides → both passed QA → 26 tissue
patches extracted → split 18/4/4. Confirms the pipeline runs end-to-end and is
ready for the real CATCH slides.

---

## 4. Artefacts produced (shown in this meeting)

| Artefact | Location |
|----------|----------|
| Before/after stain normalisation figures | `outputs/figures/stain_normalization_*.png` |
| Sample extracted patches (grids) | `outputs/figures/patches_*.png` |
| Quality assessment report | `outputs/logs/quality_report.csv` |
| Train/val/test split summary | `outputs/logs/split_summary.json` |
| Source code (modular package) | `src/` |

---

## 5. Risks & issues (current)

| Risk | Status | Mitigation in place |
|------|--------|---------------------|
| CATCH download (TCIA account / large files) | Active | Pipeline validated on synthetic data; runs unchanged on real slides once downloaded. NBIA Data Retriever workflow documented. |
| GPU resources for upcoming training | Watching | Google Colab Pro + university HPC identified as fallback (proposal risk table). |
| Class imbalance across tumour subtypes | Anticipated | Stratified split implemented; class-weighted loss + augmentation planned for Week 3+. |

---

## 6. Plan for next fortnight (Weeks 3–5)

- Complete CATCH download and run the validated pipeline on all real slides.
- Finalise patch dataset and class distribution statistics.
- **Begin U-Net segmentation model** (baseline → ResNet-34 encoder), with
  TensorBoard monitoring of loss convergence.

---

## 7. Questions for supervisor

1. Is the QA exclusion threshold (≥10% tissue, blur ≥ 50) appropriate, or should
   it be agreed with a veterinary pathologist?
2. Confirm the final list of tumour subtypes to include as classification classes.
3. Any preference on patch magnification priority (5×/10×/20×) for the first
   segmentation experiments?
