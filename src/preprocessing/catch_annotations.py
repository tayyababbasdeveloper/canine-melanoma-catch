"""CATCH polygon-annotation handling.

The real CATCH dataset ships 12,424 polygon annotations for its 350 WSIs, in two
equivalent forms (Wilm et al., 2022):

  * **MS-COCO JSON** — the portable, standard form. ``images`` carry a
    ``file_name`` (the slide), ``categories`` name each tissue/tumour class, and
    ``annotations`` give polygon ``segmentation`` rings in **level-0 pixel
    coordinates**.
  * **SQLite3** (SlideRunner schema) — the original annotation database.

This module turns those polygons into the **binary tumour masks** the U-Net needs:
for any slide it returns the list of tumour polygons, and rasterises just the part
overlapping a given tile (so it composes tile-by-tile with :mod:`src.preprocessing.wsi`
and never builds a full-slide mask in memory).

Everything else (epidermis, dermis, subcutis, inflammation/necrosis, background,
bone, cartilage) is treated as non-tumour (0); the tumour class names are taken
from ``config.catch.tumour_annotation_classes`` so the same code does binary
"tumour vs. rest" segmentation regardless of subtype.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import cv2


def find_annotation_file(search_dir: str | Path, globs) -> Path | None:
    """Return the first annotation file under ``search_dir`` matching any glob."""
    search_dir = Path(search_dir)
    for pattern in globs:
        hits = sorted(search_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None


def _polys_from_coco_segmentation(seg) -> list[np.ndarray]:
    """COCO ``segmentation`` -> list of (N,2) float arrays (polygon rings)."""
    polys = []
    if isinstance(seg, list):                      # polygon form [[x1,y1,...], ...]
        for ring in seg:
            if len(ring) >= 6:                     # at least 3 points
                polys.append(np.asarray(ring, dtype=np.float64).reshape(-1, 2))
    # RLE form (dict) is not used by CATCH; ignored on purpose.
    return polys


def load_coco_annotations(coco_path: str | Path,
                          tumour_class_names) -> dict[str, list]:
    """Parse a CATCH COCO file into ``{slide_stem: [(is_tumour, poly_xy), ...]}``.

    ``slide_stem`` is the slide file name without extension, so it matches the
    ``.svs`` files regardless of directory.
    """
    data = json.loads(Path(coco_path).read_text())
    cat_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    tumour_ids = {cid for cid, name in cat_name.items()
                  if name in set(tumour_class_names)}
    img_stem = {im["id"]: Path(im["file_name"]).stem for im in data.get("images", [])}

    per_slide: dict[str, list] = {}
    for ann in data.get("annotations", []):
        stem = img_stem.get(ann.get("image_id"))
        if stem is None:
            continue
        is_tumour = ann.get("category_id") in tumour_ids
        for poly in _polys_from_coco_segmentation(ann.get("segmentation", [])):
            per_slide.setdefault(stem, []).append((is_tumour, poly))
    return per_slide


def load_sqlite_annotations(db_path: str | Path,
                            tumour_class_names) -> dict[str, list]:
    """Best-effort parse of the CATCH SQLite (SlideRunner) annotation database.

    The COCO file is preferred; this is a fallback. The schema stores polygon
    vertices in ``Annotations_coordinates`` (coordinateX/Y, annoId, slide),
    classes in ``Classes`` (uid, name), and the per-annotation class in
    ``Annotations`` (uid, agreedClass, slide). Returns the same structure as
    :func:`load_coco_annotations`.
    """
    import sqlite3
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    try:
        classes = {uid: name for uid, name in
                   cur.execute("SELECT uid, name FROM Classes")}
        slides = {uid: Path(filename).stem for uid, filename in
                  cur.execute("SELECT uid, filename FROM Slides")}
        ann_class = {uid: (cls, slide) for uid, cls, slide in
                     cur.execute("SELECT uid, agreedClass, slide FROM Annotations")}
        coords: dict[int, list] = {}
        for anno_id, x, y in cur.execute(
                "SELECT annoId, coordinateX, coordinateY FROM Annotations_coordinates "
                "ORDER BY annoId, orderIdx"):
            coords.setdefault(anno_id, []).append((x, y))
    except sqlite3.OperationalError as exc:  # pragma: no cover - schema variance
        con.close()
        raise ValueError(f"Unexpected CATCH SQLite schema in {db_path}: {exc}")
    con.close()

    tumour = set(tumour_class_names)
    per_slide: dict[str, list] = {}
    for anno_id, pts in coords.items():
        cls_uid, slide_uid = ann_class.get(anno_id, (None, None))
        stem = slides.get(slide_uid)
        if stem is None or len(pts) < 3:
            continue
        is_tumour = classes.get(cls_uid) in tumour
        per_slide.setdefault(stem, []).append(
            (is_tumour, np.asarray(pts, dtype=np.float64)))
    return per_slide


def load_annotations(ann_path: str | Path, tumour_class_names) -> dict[str, list]:
    """Load annotations from a COCO ``.json`` or an SQLite database by extension."""
    ann_path = Path(ann_path)
    if ann_path.suffix.lower() in (".json",):
        return load_coco_annotations(ann_path, tumour_class_names)
    return load_sqlite_annotations(ann_path, tumour_class_names)


def rasterise_tile_mask(polygons, x0: int, y0: int, span0: int,
                        patch_size: int) -> np.ndarray:
    """Rasterise tumour polygons overlapping a tile into a binary patch mask.

    ``polygons`` is the per-slide list from :func:`load_coco_annotations`. The
    tile covers level-0 region ``[x0, x0+span0) x [y0, y0+span0)``; coordinates
    are shifted into the tile and scaled to ``patch_size`` so the mask aligns with
    the image patch produced by :func:`src.preprocessing.wsi.iter_tiles`.
    """
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    if not polygons:
        return mask
    scale = patch_size / float(span0)
    x1, y1 = x0 + span0, y0 + span0
    for is_tumour, poly in polygons:
        if not is_tumour:
            continue
        # quick reject: skip polygons whose bounding box misses this tile
        px_min, py_min = poly.min(axis=0)
        px_max, py_max = poly.max(axis=0)
        if px_max < x0 or px_min > x1 or py_max < y0 or py_min > y1:
            continue
        local = (poly - np.array([x0, y0])) * scale
        cv2.fillPoly(mask, [np.round(local).astype(np.int32)], 1)
    return mask


def slide_has_tumour(polygons) -> bool:
    return any(is_tumour for is_tumour, _ in (polygons or []))
