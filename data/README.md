# Data directory

The CATCH whole-slide images are **not** stored in this repository (too large, and
governed by TCIA's data-use terms). This folder only holds the structure.

```
data/
├── raw/         # Original CATCH whole-slide images (.svs / .tiff) from TCIA
├── interim/     # Macenko stain-normalised slides
└── processed/
    ├── patches/ # Extracted tissue patches (organised by class)
    └── splits/  # train.csv / val.csv / test.csv
```

## How to download the CATCH dataset

The **CAnine CuTaneous Cancer Histology (CATCH)** dataset (Fragoso-Garcia et al., 2023)
is hosted on The Cancer Imaging Archive (TCIA):

> https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941773

1. Create a free TCIA account and accept the data-use agreement.
2. Download the **NBIA Data Retriever** (TCIA's official downloader).
3. Download the `.tcia` manifest from the CATCH collection page.
4. Open the manifest in NBIA Data Retriever and save slides to `data/raw/`.

`src/data_acquisition/download_catch.py` documents this process and verifies the
downloaded files (count, file integrity).

## Dataset summary (from the proposal)

- **750** whole-slide images of canine cutaneous tumours
- Expert-annotated tumour regions, reviewed by board-certified veterinary pathologists
- Multiple subtypes: melanocytic tumours, mast cell tumours, squamous cell carcinomas, etc.
