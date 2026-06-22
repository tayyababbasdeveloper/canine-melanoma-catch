# canine-melanoma-catch

Segmentation and classification of malignant melanoma in canine cutaneous
histopathology (CATCH dataset) — MSc project **DAIM2025A_088**, University of Hull.

## Pipeline

**Phase 1 — Preprocessing (Weeks 1–2)**
```bash
python scripts/run_week1_2_pipeline.py --demo      # synthetic slides
python scripts/run_week1_2_pipeline.py --input data/raw   # real CATCH slides
```
Quality assessment → Macenko stain normalisation → multi-magnification (5×/10×/20×)
patch extraction → stratified split.

**Phase 2 — U-Net segmentation (Weeks 3–4)**
```bash
# fast end-to-end check (CPU friendly):
python scripts/run_week3_4_pipeline.py --demo --quick
# full demo run:
python scripts/run_week3_4_pipeline.py --demo
# monitor training:
tensorboard --logdir outputs/tensorboard
```
Prepare paired image+mask patches → train baseline U-Net → train ResNet-34-encoder
U-Net (transfer learning) → evaluate (Dice/IoU/pixel-acc/Hausdorff) → figures.

Individual stages:
```bash
python scripts/prepare_segmentation_data.py --demo --n-slides 8
python scripts/train_unet.py --arch unet            # ResNet-34 encoder
python scripts/train_unet.py --arch baseline        # from-scratch baseline
python scripts/make_deliverables_week3_4.py         # supervisor docx + pdf
```

## Layout
```
config/config.yaml          tunable parameters (QA, Macenko, patches, segmentation)
src/preprocessing/          tissue, stain norm, multi-mag patch extraction, split
src/data/                   augmentation, segmentation Dataset
src/models/                 U-Net (baseline + ResNet-34), losses, metrics
src/training/               trainer (cosine annealing, TensorBoard), evaluation
scripts/                    pipeline + training + deliverable entry points
docs/                       proposal, literature, progress reports, slides
```

## Setup

**Fresh Windows machine?** Follow the step-by-step guide: **[`SETUP_WINDOWS.md`](SETUP_WINDOWS.md)**.
One-command setup:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```
Manual:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```
The CATCH whole-slide images are **not** committed (large, TCIA-governed); see
`data/README.md` for the download workflow.

## Documentation
- **[`SETUP_WINDOWS.md`](SETUP_WINDOWS.md)** — full install & run guide for a new Windows system.
- **[`WHAT_TO_TRANSFER.md`](WHAT_TO_TRANSFER.md)** — which files to copy to another machine (vs. auto-generated).
- **`docs/video_guide_weeks1-4.html`** — visual walkthrough of Weeks 1–4 (open in a browser).
- `docs/Research_Proposal_DAIM2025A_088.html` — the research proposal.
- `docs/progress_report/` — fortnightly progress reports (Weeks 1–2, 3–4).
