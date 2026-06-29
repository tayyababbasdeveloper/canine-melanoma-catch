"""CATCH dataset acquisition helper (TCIA / NBIA).

The CATCH whole-slide images are distributed through The Cancer Imaging Archive
(TCIA) under a data-use agreement and must be retrieved with the official tooling
after that agreement is accepted. This module does **not** bypass the DUA. It:

  1. prints the exact, current download procedure (DOI 10.7937/TCIA.2M93-FX66);
  2. *attempts* an automated pull with the `nbiatoolkit` PyPI client if a TCIA
     manifest (``.tcia``) is present and the package is installed; and
  3. verifies / sizes the slides and annotation files placed in ``data/raw/``.

Usage:
    python -m src.data_acquisition.download_catch            # instructions + verify
    python -m src.data_acquisition.download_catch --manifest catch.tcia
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger

WSI_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")
ANN_EXTENSIONS = (".json", ".sqlite", ".sqlite3", ".db")

DOI = "10.7937/TCIA.2M93-FX66"
TCIA_URL = "https://www.cancerimagingarchive.net/collection/catch/"

INSTRUCTIONS = f"""
================ CATCH dataset download (TCIA) ================
Reference : Wilm et al. (2022) 'Pan-tumor CAnine cuTaneous Cancer Histology
            (CATCH) dataset', Scientific Data 9, 588.
DOI       : {DOI}
Collection: {TCIA_URL}
Contents  : 350 .svs WSIs (50 each of 7 tumour subtypes) + 12,424 polygon
            annotations (MS-COCO JSON + SQLite3).

1. Create a free TCIA account and accept the CATCH data-use agreement.
2. On the collection page, download the '.tcia' manifest (images) and the
   supplemental annotation archive (COCO JSON + SQLite).
3. Get the NBIA Data Retriever (GUI) OR install the Python client:
       pip install nbiatoolkit
4a. GUI : open the manifest in NBIA Data Retriever, save slides into data/raw/.
4b. CLI : python -m src.data_acquisition.download_catch --manifest <file>.tcia
5. Unzip the annotation archive so the COCO .json (and/or .sqlite) sits under
   data/raw/ (any subfolder — it is auto-discovered).
6. Re-run this script (no args) to verify the download.

Recommended on-disk layout (subtype folders make labelling unambiguous):
    data/raw/
      Melanoma/*.svs   Mast cell tumor/*.svs   Squamous cell carcinoma/*.svs
      Peripheral nerve sheath tumor/*.svs   Trichoblastoma/*.svs
      Histiocytoma/*.svs   Plasmacytoma/*.svs
      annotations/CATCH.json        (MS-COCO polygons)
==============================================================
"""


def try_nbia_download(manifest: Path, raw_dir: Path, logger) -> bool:
    """Attempt an automated pull via nbiatoolkit. Returns True on success."""
    try:
        from nbiatoolkit import NBIAClient  # type: ignore
    except ImportError:
        logger.warning("nbiatoolkit not installed (`pip install nbiatoolkit`); "
                       "use the NBIA Data Retriever GUI with %s instead.", manifest)
        return False
    if not manifest.exists():
        logger.error("Manifest not found: %s", manifest)
        return False
    try:
        logger.info("Downloading CATCH via nbiatoolkit from %s ...", manifest)
        series = [ln.strip() for ln in manifest.read_text().splitlines()
                  if ln.strip() and not ln.startswith("#") and "." in ln]
        client = NBIAClient()
        for uid in series:
            client.downloadSeries(uid, str(raw_dir))
        logger.info("nbiatoolkit download finished.")
        return True
    except Exception as exc:  # pragma: no cover - network/auth dependent
        logger.error("Automated download failed (%s). Fall back to the GUI.", exc)
        return False


def verify_download(raw_dir: Path, logger) -> dict:
    """List/size the slides and locate annotation files in ``data/raw/``."""
    slides = sorted(p for p in raw_dir.rglob("*")
                    if p.suffix.lower() in WSI_EXTENSIONS
                    and "demo" not in p.stem and p.parent.name != "masks")
    anns = sorted(p for p in raw_dir.rglob("*")
                  if p.suffix.lower() in ANN_EXTENSIONS)

    if not slides:
        logger.warning("No CATCH whole-slide images found in %s", raw_dir)
        logger.info(INSTRUCTIONS)
        return {"slides": 0, "annotations": len(anns)}

    total_gb = sum(s.stat().st_size for s in slides) / (1024 ** 3)
    logger.info("Found %d WSI(s) in %s (%.2f GB; target: 350 slides)",
                len(slides), raw_dir, total_gb)
    # report subtype folders if the recommended layout was used
    subtypes = sorted({s.parent.name for s in slides
                       if s.parent.name not in ("raw", "images")})
    if subtypes:
        logger.info("Subtype folders present: %s", subtypes)
    if anns:
        logger.info("Annotation file(s): %s", [a.name for a in anns])
    else:
        logger.warning("No annotation file (COCO .json / .sqlite) found — "
                       "segmentation masks cannot be built without it.")
    return {"slides": len(slides), "annotations": len(anns),
            "size_gb": round(total_gb, 2), "subtypes": subtypes}


def main() -> None:
    ap = argparse.ArgumentParser(description="CATCH download / verify (TCIA)")
    ap.add_argument("--manifest", default=None,
                    help="TCIA .tcia manifest for an automated nbiatoolkit pull")
    args = ap.parse_args()

    cfg = load_config()
    logger = get_logger("download_catch", cfg["paths"]["logs_dir"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        try_nbia_download(Path(args.manifest), raw_dir, logger)
    verify_download(raw_dir, logger)


if __name__ == "__main__":
    main()
