"""Build the SELF-CONTAINED project notebook.

Per the supervisor's requirement that *all code lives in the notebook*, this script
inlines every ``src/`` module and the config as runnable cells, so the notebook
defines the entire project (preprocessing, WSI/annotation ingestion, models,
training, evaluation, demo generators) with **no imports from ``src``** — then adds
the narrative, the real-CATCH-annotation section, a live preprocessing demo, and
the saved results.

Run:  python scripts/build_full_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "CATCH_Melanoma_Project.ipynb"


# --------------------------------------------------------------------------- #
# cell helpers
# --------------------------------------------------------------------------- #
def md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def inline_module(rel_path: str) -> str:
    """Read a src module and make it notebook-safe:
    - drop ``from __future__ import annotations`` (cells aren't modules),
    - drop internal ``from src...`` / ``import src`` imports (incl. multi-line),
      since every name is already defined by an earlier cell,
    - map ``Path(__file__).resolve().parents[2]`` -> the notebook's ``ROOT``.
    External imports (numpy, cv2, torch, ...) are kept; re-importing is harmless.
    """
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    out, skip_paren = [], False
    for line in text.split("\n"):
        if skip_paren:
            if ")" in line:
                skip_paren = False
            continue
        s = line.lstrip()
        if s.startswith("from __future__"):
            continue
        if s.startswith("from src") or s.startswith("import src"):
            if "(" in line and ")" not in line:
                skip_paren = True
            continue
        # Neutralise headless-backend switches so the notebook plots inline
        # (savefig still works under the inline backend).
        if s.replace(" ", "").startswith('matplotlib.use("Agg")') or \
           s.replace(" ", "").startswith("matplotlib.use('Agg')"):
            indent = line[:len(line) - len(s)]
            out.append(indent + "pass  # matplotlib.use('Agg') removed for inline plots")
            continue
        out.append(line)
    body = "\n".join(out)
    body = body.replace("Path(__file__).resolve().parents[2]", "ROOT")
    body = body.replace("Path(__file__).resolve().parents[1]", "ROOT")
    return body.strip("\n")


def module_cell(rel_path: str, title: str, purpose: str) -> list:
    """Markdown header + code cell for one inlined module."""
    return [
        md(f"#### `{rel_path}` — {title}\n{purpose}"),
        code(inline_module(rel_path)),
    ]


cells: list = []

# =========================================================================== #
# Title + plan
# =========================================================================== #
cells.append(md(r"""
# 🐶 Classification of Malignant Melanoma in Canines (CATCH) — **Full Code Notebook**
### Weeks 1–6 · Segmentation & Classification · Project DAIM2025A_088

**Student:** Muhammad Tayyab Abbas &nbsp;·&nbsp; **Supervisor:** Dr Claire Cashmore &nbsp;·&nbsp; University of Hull

This is the **self-contained** project notebook: **every module of the project's
source code is defined inline here** (no hidden `src/` imports), followed by the
real CATCH data, a live preprocessing demo, and the results. Run the cells top to
bottom and the whole project is defined and exercised inside this one file.

**Contents**
1. Setup & configuration (inline)
2. **The full source code**, module by module (preprocessing → models → training)
3. Real CATCH dataset — folder & annotations
4. Live preprocessing demo (Weeks 1–2)
5. Segmentation & classification results (Weeks 3–6)
6. How to run on the full real dataset
"""))

cells.append(md(r"""
## 0. Plan — segmentation → classification (the supervisor's questions)

```
Whole-slide image → 256×256 patches (5×/10×/20×)
   → [U-Net SEGMENTATION] → per-pixel tumour mask
       → keep tumour patches → [CNN CLASSIFICATION] → tumour SUBTYPE
```

- **Segmentation output:** a binary tumour mask per patch (`1`=tumour, `0`=rest).
- **Classification:** the tumour **subtype** — the real **7 CATCH subtypes**
  (Melanoma, Mast Cell Tumor, SCC, PNST, Trichoblastoma, Histiocytoma,
  Plasmacytoma), or binary melanoma-vs-rest.
- **Imbalance:** segmentation (tumour ≈ 9% of pixels) → BCE `pos_weight` + Dice;
  subtype imbalance → class-weighted cross-entropy, **after** segmentation.
"""))

# =========================================================================== #
# 1. Setup & imports
# =========================================================================== #
cells.append(md(r"""
## 1. Setup, imports & configuration

The first cell sets up the project root and all external libraries. The second
embeds the project configuration (identical to `config/config.yaml`) inline.
"""))

cells.append(code(r"""
# --- external libraries used across the whole project ---
import os, sys, json, time, math, warnings, sqlite3, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# project root = the folder that contains config/config.yaml
ROOT = Path.cwd()
while not (ROOT / "config" / "config.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
print("Project root:", ROOT)
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())

FIG = ROOT / "outputs" / "figures"
LOGS = ROOT / "outputs" / "logs"

def show_img(path, title=None, size=(6, 6)):
    p = Path(path)
    if not p.exists():
        print("(not found — run the pipeline to create:", p.name, ")"); return
    import matplotlib.image as mpimg
    plt.figure(figsize=size); plt.imshow(mpimg.imread(str(p))); plt.axis("off")
    if title: plt.title(title)
    plt.show()

def load_json(name):
    p = LOGS / name
    return json.loads(p.read_text()) if p.exists() else None
"""))

# ---- config inline (verbatim copy of config/config.yaml) ----
_cfg_yaml = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
cfg_cell = (
    "import yaml\n\n"
    "# The project configuration, embedded verbatim from config/config.yaml so the\n"
    "# notebook is fully self-contained.\n"
    "_CONFIG_YAML = r'''\n" + _cfg_yaml + "\n'''\n\n"
    "PROJECT_ROOT = ROOT\n\n"
    "def load_config():\n"
    "    cfg = yaml.safe_load(_CONFIG_YAML)\n"
    "    for k, v in cfg.get('paths', {}).items():\n"
    "        cfg['paths'][k] = (PROJECT_ROOT / v).resolve()\n"
    "    for section in ('segmentation', 'classification'):\n"
    "        sub = cfg.get(section, {}).get('paths', {})\n"
    "        for k, v in sub.items():\n"
    "            sub[k] = (PROJECT_ROOT / v).resolve()\n"
    "    return cfg\n\n"
    "def ensure_dirs(cfg):\n"
    "    for key in ('interim_dir','processed_dir','patches_dir','figures_dir','logs_dir'):\n"
    "        p = cfg['paths'].get(key)\n"
    "        if p is not None: Path(p).mkdir(parents=True, exist_ok=True)\n"
    "    for section in ('segmentation','classification'):\n"
    "        for p in cfg.get(section, {}).get('paths', {}).values():\n"
    "            if p is not None: Path(p).mkdir(parents=True, exist_ok=True)\n\n"
    "cfg = load_config(); ensure_dirs(cfg)\n"
    "print('Config loaded. Real CATCH subtypes:', cfg['catch']['subtypes'])\n"
)
cells.append(code(cfg_cell))

# =========================================================================== #
# 2. The full source code — inlined module by module
# =========================================================================== #
cells.append(md(r"""
## 2. The project source code (inline)

Every module below is the **actual project code**, defined here so the notebook is
self-contained. They are ordered by dependency, so running them top-to-bottom
defines the whole library used in the rest of the notebook.
"""))

cells.append(md("### 2.1 Utilities"))
cells += module_cell("src/utils/seed.py", "reproducible seeding",
                     "Seeds Python/NumPy/PyTorch for reproducibility.")
cells += module_cell("src/utils/logger.py", "console+file logger",
                     "A small dual console/file logger used by the pipeline scripts.")

cells.append(md("### 2.2 Preprocessing"))
cells += module_cell("src/preprocessing/image_io.py", "image / WSI loading",
                     "Loads standard images, or a WSI thumbnail via OpenSlide.")
cells += module_cell("src/preprocessing/tissue.py", "tissue detection",
                     "HSV-saturation + Otsu tissue mask (drops glass background).")
cells += module_cell("src/preprocessing/stain_normalization.py", "Macenko normalisation",
                     "Macenko (2009) H&E stain normalisation in optical-density space.")
cells += module_cell("src/preprocessing/patch_extraction.py", "patch tiling",
                     "Multi-magnification image (+mask) patch extraction.")
cells += module_cell("src/preprocessing/wsi.py", "real .svs tiling",
                     "Reads real CATCH whole-slide images tile-by-tile via OpenSlide.")
cells += module_cell("src/preprocessing/catch_annotations.py", "COCO/SQLite → masks",
                     "Parses the real CATCH polygon annotations into tumour masks.")
cells += module_cell("src/preprocessing/dataset_split.py", "slide-level split",
                     "Stratified, slide-level train/val/test split (no patch leakage).")

cells.append(md("### 2.3 Data acquisition & quality"))
cells += module_cell("src/data_acquisition/quality_check.py", "slide QA",
                     "Brightness / blur / tissue-fraction quality assessment.")

cells.append(md("### 2.4 Datasets & augmentation"))
cells += module_cell("src/data/augmentation.py", "augmentation pipelines",
                     "Albumentations transforms (shared image+mask geometric, photometric).")
cells += module_cell("src/data/segmentation_dataset.py", "segmentation Dataset",
                     "PyTorch Dataset for image+mask patches.")
cells += module_cell("src/data/classification_dataset.py", "classification Dataset",
                     "PyTorch Dataset for subtype-labelled patches.")

cells.append(md("### 2.5 Models"))
cells += module_cell("src/models/losses.py", "BCE + Dice loss",
                     "Soft Dice and composite BCE+Dice losses.")
cells += module_cell("src/models/metrics.py", "segmentation metrics",
                     "Micro-averaged Dice / IoU / pixel-acc / Hausdorff.")
cells += module_cell("src/models/unet.py", "baseline & ResNet-34 U-Net",
                     "From-scratch U-Net and a ResNet-34-encoder U-Net (smp).")
cells += module_cell("src/models/attention_unet.py", "Attention U-Net",
                     "Attention U-Net (Oktay 2018) on the baseline backbone.")
cells += module_cell("src/models/classifier.py", "ResNet-50 / EfficientNet-B3",
                     "Transfer-learning classifier with progressive unfreezing + frozen-BN fix.")

cells.append(md("### 2.6 Training & evaluation"))
cells += module_cell("src/training/trainer.py", "segmentation trainer",
                     "Adam + cosine LR + AMP + early stopping + TensorBoard.")
cells += module_cell("src/training/classifier_trainer.py", "classification trainer",
                     "Class-weighted CE + progressive unfreezing + metrics.")
cells += module_cell("src/training/evaluate.py", "evaluation & figures",
                     "Test aggregation and prediction/curve figures.")

cells.append(md("### 2.7 Synthetic demo-data generators (pipeline-check only)"))
cells += module_cell("src/utils/demo_data.py", "demo H&E slides",
                     "Synthetic H&E slides (Week 1–2 preprocessing demo).")
cells += module_cell("src/utils/demo_segmentation.py", "demo slides + masks",
                     "Synthetic slides with ground-truth tumour masks (segmentation).")
cells += module_cell("src/utils/demo_classification.py", "demo subtype slides",
                     "Synthetic colour-separable subtype slides (classification).")

cells.append(md(r"""
### 2.8 Quick sanity-check — the inlined code actually runs

Build each model and run a forward pass, and compute the losses/metrics, to prove
the inlined definitions above are correct and self-consistent.
"""))
cells.append(code(r"""
x = torch.randn(2, 3, 256, 256)
base = BaselineUNet(in_channels=3, classes=1)
attn = AttentionUNet(in_channels=3, classes=1)
clf  = build_classifier("resnet50", num_classes=7, pretrained=False)
print("BaselineUNet  out:", tuple(base(x).shape),  "| params %.2fM" % (sum(p.numel() for p in base.parameters())/1e6))
print("AttentionUNet out:", tuple(attn(x).shape),  "| params %.2fM" % (sum(p.numel() for p in attn.parameters())/1e6))
print("ResNet-50     out:", tuple(clf(x).shape),   "| params %.2fM" % (sum(p.numel() for p in clf.parameters())/1e6))

# losses + metrics on a toy batch
logits = base(x); target = (torch.rand(2,1,256,256) > 0.5).float()
print("Dice loss:", round(float(DiceLoss()(logits, target)), 4))
m = SegMetrics(); m.update(logits, target)
print("metrics:", {k: round(v,4) for k,v in m.compute().items() if k!='hausdorff'})
print("All inlined models, losses and metrics run correctly.")
"""))

# =========================================================================== #
# 3. Real CATCH data
# =========================================================================== #
cells.append(md(r"""
## 3. The real CATCH dataset — folder & annotations

The **real annotations are downloaded** (`data/raw/annotations/`), so the cells
below show the genuine 7-class distribution and real tumour polygons. The 522 GB of
whole-slide images do not fit on this machine, so model training (§5) uses synthetic
slides as a pipeline check — the code is identical for real slides (§6).
"""))

cells.append(code(r"""
raw = ROOT / "data" / "raw"
print("data/raw/ structure:\n")
for sub in cfg["catch"]["subtypes"]:
    d = raw / sub
    n = len(list(d.glob("*.svs"))) if d.exists() else 0
    print(f"  {sub+'/':<22} {n:>3} .svs   (target: 50)")
for f in ["CATCH.json", "CATCH.sqlite"]:
    p = raw / "annotations" / f
    print(f"  annotations/{f:<16}", f"{p.stat().st_size/1e6:.0f} MB" if p.exists()
          else "MISSING (run: python -m src.data_acquisition.download_catch)")
"""))

cells.append(code(r"""
# REAL class distribution + a REAL tumour polygon, straight from CATCH.json
catch = cfg["catch"]
ann_file = find_annotation_file(raw, catch["coco_annotation_glob"])
if ann_file is None:
    print("Annotations not downloaded yet. Run: python -m src.data_acquisition.download_catch")
else:
    stats = coco_dataset_stats(ann_file, catch["tumour_annotation_classes"])
    print(f"REAL CATCH: {stats['n_slides']} slides, {stats['n_annotations']} polygons")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    spp = stats["slides_per_subtype"]
    ax[0].bar(spp.keys(), spp.values(), color="#1F3A6E")
    ax[0].set_title("Real slides per subtype (350 total)"); ax[0].tick_params(axis="x", rotation=40)
    apc = stats["annotations_per_class"]
    ax[1].bar(apc.keys(), apc.values(), color="#8E44AD")
    ax[1].set_title("Real annotations per class"); ax[1].tick_params(axis="x", rotation=60)
    plt.tight_layout(); plt.show()

    per_slide = load_coco_annotations(ann_file, catch["tumour_annotation_classes"])
    sid = next(k for k in per_slide if k.startswith("Melanoma"))
    tum = [p for is_t, p in per_slide[sid] if is_t]
    c = tum[0].mean(0).astype(int); span = 4000
    mask = rasterise_tile_mask(per_slide[sid], int(c[0]-span/2), int(c[1]-span/2), span, 512)
    plt.figure(figsize=(4.5,4.5)); plt.imshow(mask, cmap="magma")
    plt.title(f"Real tumour polygon → mask\n{sid}"); plt.axis("off"); plt.show()
"""))

# =========================================================================== #
# 4. Live preprocessing demo
# =========================================================================== #
cells.append(md(r"""
## 4. Weeks 1–2 — live preprocessing demo

Using the inlined code above: generate a synthetic H&E slide, run **Macenko**
normalisation, **tissue** detection, and **multi-magnification** patch extraction.
"""))
cells.append(code(r"""
img, mask = generate_he_slide_with_mask(size=1024, n_tumours=2, seed=7)
norm = MacenkoNormalizer(**cfg["stain_normalization"]).normalize(img)
tm = tissue_mask(img)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, im, t, cm in [(ax[0],img,"1. Original H&E",None),(ax[1],norm,"2. Macenko",None),
                     (ax[2],tm,"3. Tissue mask","gray"),(ax[3],mask,"4. Tumour mask","gray")]:
    a.imshow(im, cmap=cm); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()

counts = {}
for (mg, y, x, p, mp) in extract_multimag_with_mask(norm, mask, [5,10,20], 20, 256, 256, 0.5, 2000, tissue_img=img):
    counts[mg] = counts.get(mg, 0) + 1
print("patches per magnification:", counts)
"""))

# =========================================================================== #
# 5. Results
# =========================================================================== #
cells.append(md(r"""
## 5. Weeks 3–6 — results (loaded from the saved runs)

Segmentation (3-way) and classification results from the trained runs. (Training
uses the inlined trainers above; the full commands are in §6.)
"""))
cells.append(code(r"""
seg3 = load_json("week5_6_seg_comparison.json")
if seg3:
    display(pd.DataFrame(seg3).T.round(3))
show_img(FIG/"unet_attention_predictions.png", "Attention U-Net — test predictions", (9,9))

clsrep = load_json("cls_resnet50_test_report.json")
if clsrep:
    m = clsrep["test_metrics"]
    print("Classification classes:", clsrep["classes"])
    print("TEST  acc=%.3f  macro-F1=%.3f  AUC=%s" % (
        m["accuracy"], m["f1"], "n/a" if m["auc_roc"]!=m["auc_roc"] else round(m["auc_roc"],3)))
show_img(FIG/"cls_resnet50_confusion.png", "ResNet-50 — confusion matrix", (5,5))
"""))

# =========================================================================== #
# 6. Real run + summary
# =========================================================================== #
cells.append(md(r"""
## 6. Running on the full real CATCH dataset

The same inlined code runs on the real slides. Place the `.svs` files under
`data/raw/<Subtype>/` (or flat — labels come from the filename prefix, e.g.
`MCT_15_1.svs`), then:

```bash
python -m src.data_acquisition.download_catch              # annotations + folders
python scripts/prepare_segmentation_data.py  --input data/raw
python scripts/train_unet.py --arch attention
python scripts/prepare_classification_data.py --input data/raw
python scripts/train_classifier.py --arch resnet50 --epochs 20
```

> Needs ~600 GB free + OpenSlide binaries. The 522 GB of WSIs download via the
> Aspera plugin from the TCIA collection page (DOI 10.7937/TCIA.2M93-FX66).

## 7. Summary

**Done (W1–6):** preprocessing · baseline + ResNet-34 + **Attention** U-Net
segmentation · **ResNet-50** subtype classification · real CATCH annotations
integrated · **all source code now lives in this notebook**.

**Next:** EfficientNet-B3 comparison · real-slide run on a larger disk · evaluation
(5-fold CV, McNemar's test, Grad-CAM).
"""))

# --------------------------------------------------------------------------- #
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Self-contained notebook written -> {OUT}")
print(f"  {len(cells)} cells ({n_code} code) — all source inlined")
