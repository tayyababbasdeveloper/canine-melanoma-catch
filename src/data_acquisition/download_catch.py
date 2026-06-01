"""CATCH dataset acquisition helper.

The CATCH whole-slide images are distributed through The Cancer Imaging Archive
(TCIA) and must be downloaded with the official NBIA Data Retriever after
accepting the data-use agreement. This module does NOT bypass that process; it
documents it and then *verifies* the slides that have been placed in data/raw/.

Usage:
    python -m src.data_acquisition.download_catch
"""
from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger

# Slide file extensions used by the CATCH collection / TCIA
WSI_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")

TCIA_URL = ("https://wiki.cancerimagingarchive.net/pages/viewpage.action"
            "?pageId=101941773")

INSTRUCTIONS = f"""
================ CATCH dataset download instructions ================
1. Create a free TCIA account and accept the data-use agreement.
2. Visit the CATCH collection page:
       {TCIA_URL}
3. Download the official NBIA Data Retriever tool.
4. Download the collection's `.tcia` manifest file.
5. Open the manifest in NBIA Data Retriever and save the slides into:
       data/raw/
6. Re-run this script to verify the download.
====================================================================
"""


def verify_download(raw_dir: Path, logger) -> list[Path]:
    """List and sanity-check the slides present in data/raw/."""
    slides = sorted(
        p for p in raw_dir.rglob("*") if p.suffix.lower() in WSI_EXTENSIONS
    )
    if not slides:
        logger.warning("No whole-slide images found in %s", raw_dir)
        logger.info(INSTRUCTIONS)
        return []

    logger.info("Found %d whole-slide image(s) in %s", len(slides), raw_dir)
    total_mb = 0.0
    for s in slides:
        size_mb = s.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        if size_mb < 0.01:
            logger.warning("  [!] %s is suspiciously small (%.3f MB)", s.name, size_mb)
    logger.info("Total size: %.1f MB across %d slides "
                "(target: 750 slides)", total_mb, len(slides))
    return slides


def main() -> None:
    cfg = load_config()
    logger = get_logger("download_catch", cfg["paths"]["logs_dir"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    verify_download(raw_dir, logger)


if __name__ == "__main__":
    main()
