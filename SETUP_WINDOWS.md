# Setup & Run Guide — Windows (Step by Step)

This guide takes a **fresh Windows 10/11 machine** from nothing to running the full
canine-melanoma pipeline (Weeks 1–4). Follow it top to bottom.

> TL;DR (if you just want it running):
> ```powershell
> git clone https://github.com/Dilawarkhaninfo/canine-melanoma-catch.git
> cd canine-melanoma-catch
> powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
> .\.venv\Scripts\Activate.ps1
> python scripts\run_week1_2_pipeline.py --demo
> python scripts\run_week3_4_pipeline.py --demo --quick
> ```

---

## 1. Prerequisites (install these once)

| Tool | Version | Where | Notes |
|------|---------|-------|-------|
| **Python** | 3.10 – 3.14 (3.12 recommended) | https://www.python.org/downloads/ | **Tick "Add python.exe to PATH"** during install |
| **Git** | latest | https://git-scm.com/download/win | needed to clone the repo |
| **Git LFS** | latest | https://git-lfs.com | needed to download the trained model files (`.pt`). Install it **before** cloning, then run `git lfs install` |
| **OpenSlide** (optional) | latest | https://openslide.org/download/ | only needed for REAL `.svs/.tiff` slides; the demo does NOT need it |

Check Python is on PATH (open a new **PowerShell** window):
```powershell
python --version
```
You should see e.g. `Python 3.12.x`. If "not recognised", reinstall Python with the
PATH checkbox ticked, or use `py --version`.

---

## 2. Get the code

```powershell
# install Git LFS once (so the trained .pt models come down with the clone):
git lfs install
cd C:\Users\<you>\Documents          # or wherever you keep projects
git clone https://github.com/Dilawarkhaninfo/canine-melanoma-catch.git
cd canine-melanoma-catch
git lfs pull                         # ensures the model files are real (not pointers)
```

> The repo now ships with the **trained models, figures, predictions and demo
> patches** committed (models via Git LFS). So after the clone you already have
> the **identical results** — you don't have to retrain. If you skipped Git LFS,
> the `.pt` files will be tiny pointer stubs; install Git LFS and run
> `git lfs pull` to fetch the real ones.

---

## 3. Set up the environment (one command)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

This script:
1. creates an isolated virtual environment in `.venv`,
2. installs the **CPU build** of PyTorch (works on any machine, no GPU needed),
3. installs every other dependency from `requirements.txt`,
4. prints a verification line ending in `OK  torch ...`.

**Have a CUDA GPU?** install the GPU build instead:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Gpu
```

### Manual alternative (if the script fails)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CPU PyTorch first (important - gets the right build):
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --retries 10 --timeout 180
# then everything else:
python -m pip install -r requirements.txt --retries 10 --timeout 180
```

---

## 4. Activate the environment (every new terminal)

```powershell
.\.venv\Scripts\Activate.ps1
```
Your prompt now starts with `(.venv)`. If you get a script-execution error, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 5. Run the pipelines (demo mode — no real data needed)

### Weeks 1–2 — Preprocessing
```powershell
python scripts\run_week1_2_pipeline.py --demo
```
Produces: quality report, stain-normalisation figures, extracted patches, and a
train/val/test split under `outputs\` and `data\processed\`.

### Weeks 3–4 — U-Net segmentation
Quick end-to-end check (a few minutes on CPU):
```powershell
python scripts\run_week3_4_pipeline.py --demo --quick
```
Full demo run (more slides / epochs, ~1 hour on CPU):
```powershell
python scripts\run_week3_4_pipeline.py --demo --n-slides 14 --baseline-epochs 6 --epochs 12 --max-train 100 --max-eval 60
```

This trains a **baseline U-Net** then a **ResNet-34-encoder U-Net**, evaluates both,
and writes:
- training-curve + prediction figures → `outputs\figures\`
- metrics + comparison → `outputs\logs\*.json`
- model checkpoints → `outputs\checkpoints\`

---

## 6. View training in TensorBoard (optional)

```powershell
tensorboard --logdir outputs\tensorboard
```
Open the printed URL (usually http://localhost:6006) in a browser.

---

## 7. Generate the supervisor documents (optional)

```powershell
python scripts\make_deliverables.py            # Week 1-2 report + slides
python scripts\make_deliverables_week3_4.py    # Week 3-4 report + slides
```
Outputs land in `docs\progress_report\` (`.docx`) and `docs\slides\` (`.pdf`).

---

## 8. Running on the REAL CATCH data (later)

1. Install the **OpenSlide binaries** (Prerequisites table) and add the `bin` folder to PATH.
2. Download the CATCH slides from TCIA into `data\raw\` — see `data\README.md`.
3. For segmentation, place slides in `data\raw\images\` and masks in `data\raw\masks\`.
4. Run the pipelines **without** `--demo`:
   ```powershell
   python scripts\run_week1_2_pipeline.py --input data\raw
   python scripts\prepare_segmentation_data.py --input data\raw
   python scripts\train_unet.py --arch unet
   ```

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `python` not recognised | Reinstall Python with "Add to PATH", or use `py` instead of `python`. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then retry. |
| pip `Connection broken / reset` | Re-run the same command; pip resumes. The setup script already uses `--retries 10`. |
| `No module named torch` after install | You skipped the CPU-index step. Run the two `pip install` lines in §3 manually. |
| `Could not write train.csv` (PermissionError) | Close the CSV in Excel, then re-run. |
| huggingface `symlinks` warning during ResNet-34 | Harmless. To silence: `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1` (new terminal). |
| OpenSlide import error | Only needed for real `.svs` slides; install the OpenSlide binaries and add `bin\` to PATH. |
| Training too slow on CPU | Use `--quick`, lower `--epochs`, or a smaller `--max-train`; or set up the `-Gpu` build on a CUDA machine. |

---

## 10. What gets created (so you know it worked)

```
data\processed\seg_patches\      paired image+mask patches + manifest.csv
data\processed\seg_splits\       train/val/test manifests + summary
outputs\figures\unet_*_curves.png        training curves
outputs\figures\unet_*_predictions.png   image | ground-truth | prediction panels
outputs\logs\week3_4_comparison.json     baseline vs ResNet-34 metrics
outputs\checkpoints\*.pt                  trained model weights
```

You're set. For a visual walkthrough of what each week does, open
`docs\video_guide_weeks1-4.html` in a browser.
