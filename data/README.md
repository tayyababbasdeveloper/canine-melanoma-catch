# Data directory — where the CATCH data lives

The CATCH **whole-slide images** are **not** stored in this repository (522 GB, and
governed by TCIA's terms). The **annotations are small and public**, so they *are*
downloaded here. This file documents exactly where each piece of data goes.

```
data/
├── raw/
│   ├── annotations/              ← REAL CATCH annotations (downloaded, ~190 MB)
│   │   ├── CATCH.json            ← MS-COCO polygons (350 slides, 12,424 annotations)
│   │   └── CATCH.sqlite          ← same annotations, SlideRunner SQLite DB
│   ├── Melanoma/        *.svs    ← 50 real WSIs, named Melanoma_1_1.svs …
│   ├── Mast Cell Tumor/ *.svs    ← 50 real WSIs, named MCT_1_1.svs …
│   ├── SCC/             *.svs    ← 50 real WSIs, named SCC_1_1.svs …
│   ├── PNST/            *.svs    ← 50 real WSIs, named PNST_1_1.svs …
│   ├── Trichoblastoma/  *.svs    ← 50 real WSIs
│   ├── Histiocytoma/    *.svs    ← 50 real WSIs
│   ├── Plasmacytoma/    *.svs    ← 50 real WSIs
│   ├── images/  masks/  cls_images/   ← (demo only) synthetic slides for the pipeline demo
├── interim/                      ← Macenko stain-normalised slides
└── processed/
    ├── seg_patches/  seg_splits/ ← segmentation patches + train/val/test manifests
    ├── cls_patches/  cls_splits/ ← classification patches + manifests
    └── patches/  splits/         ← Week 1-2 preprocessing output
```

## Status — what is and isn't downloaded

| Component | Size | In repo? | How |
|-----------|------|----------|-----|
| **Annotations** (COCO + SQLite) | ~190 MB | ✅ downloaded to `data/raw/annotations/` | public, auto-downloaded |
| **Whole-slide images** (`.svs`) | **522 GB** | ❌ not present | TCIA / Aspera (see below) |

The annotations alone already give the **real 7-class distribution (50 slides each)**
and the **real tumour polygons** — these drive the notebook's real-data section
without needing the 522 GB of images.

## Downloading the slides (`.svs`)

The **CAnine cuTaneous Cancer Histology (CATCH)** dataset (Wilm et al., 2022) is on
The Cancer Imaging Archive, CC BY 4.0:

> https://www.cancerimagingarchive.net/collection/catch/ — DOI 10.7937/TCIA.2M93-FX66

1. **Annotations** (already here) were fetched automatically:
   ```bash
   python -m src.data_acquisition.download_catch        # downloads annotations + verifies
   ```
2. **Whole-slide images (522 GB)** are downloaded via the **IBM Aspera Connect**
   browser plugin from the collection page (there is no smaller per-slide API).
   Save each subtype's slides into its folder above — the filename prefix
   (`MCT_`, `Melanoma_`, `SCC_`, `PNST_`, `Trichoblastoma_`, `Histiocytoma_`,
   `Plasmacytoma_`) determines the label automatically.

   > ⚠️ Needs ~600 GB free + the **OpenSlide** binaries (`pip install openslide-python`
   > plus the platform libraries from openslide.org). This machine has only ~59 GB
   > free, so the full set must be downloaded on a larger disk.

3. Then run the real pipeline (no `--demo`):
   ```bash
   python scripts/prepare_segmentation_data.py  --input data/raw
   python scripts/train_unet.py --arch attention
   python scripts/prepare_classification_data.py --input data/raw
   python scripts/train_classifier.py --arch resnet50 --epochs 20
   ```

## The 7 tumour subtypes (exact names from the annotation files)

Melanoma · Mast Cell Tumor · SCC · PNST · Trichoblastoma · Histiocytoma · Plasmacytoma
— 50 whole-slide images each, **350 total** (12,424 polygon annotations).
