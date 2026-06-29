# CATCH annotations (real data)

These two files are the **real** CATCH polygon annotations, downloaded from TCIA
(public, CC BY 4.0). They are **not committed** to git (too large, re-downloadable)
but are fetched automatically by:

```bash
python -m src.data_acquisition.download_catch
```

| File | Size | Format | Contents |
|------|------|--------|----------|
| `CATCH.json` | ~60 MB | MS-COCO | 350 slides, 12,424 polygon annotations, 13 classes |
| `CATCH.sqlite` | ~130 MB | SlideRunner | same annotations as a relational DB |

**13 annotation classes:** Bone, Cartilage, Dermis, Epidermis, Subcutis,
Inflamm/Necrosis (tissue) + the 7 tumour subtypes Melanoma, Plasmacytoma,
Mast Cell Tumor, PNST, SCC, Trichoblastoma, Histiocytoma.

For segmentation, the 7 tumour classes are merged into a single **tumour** mask
(everything else = background). `src/preprocessing/catch_annotations.py` parses both
formats and rasterises a tumour mask per image tile.
