"""Stratified, **slide-level** train / validation / test split.

Splits patches into 70/15/15 subsets while preserving the class distribution
(proposal section 3.1.1: stratified sampling). Crucially, when a ``group_col``
(e.g. ``slide_id``) is given, the split is done at the **slide** level — every
patch of a slide stays in a single subset — so neighbouring tiles of one slide
cannot leak across train/val/test and inflate the metrics. Writes
train.csv / val.csv / test.csv.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _group_split(df: pd.DataFrame, group_col: str, label_col: str,
                 train: float, val: float, seed: int):
    """Partition unique groups (slides) into train/val/test, stratified by the
    group's (majority) label so each subset keeps a similar class mix.

    Count-based per class (robust for both tiny demos and the full 350-slide set):
    each class's slides are shuffled and split by integer counts, reserving one
    val and one test slide per class whenever the class has enough slides.
    """
    g = (df.groupby(group_col)[label_col]
           .agg(lambda s: s.value_counts().index[0]).reset_index())
    rng = np.random.default_rng(seed)
    train_s, val_s, test_s = set(), set(), set()
    for cls in sorted(g[label_col].unique()):
        slides = g[g[label_col] == cls][group_col].to_numpy()
        rng.shuffle(slides)
        n = len(slides)
        if n == 1:
            n_tr, n_va, n_te = 1, 0, 0
        elif n == 2:
            n_tr, n_va, n_te = 1, 0, 1
        else:
            n_va = max(1, int(round(val * n)))
            n_te = max(1, int(round((1.0 - train - val) * n)))
            n_tr = n - n_va - n_te
            if n_tr < 1:                      # guarantee a non-empty train set
                n_tr, n_te = 1, n - 1 - n_va
        train_s |= set(slides[:n_tr])
        val_s |= set(slides[n_tr:n_tr + n_va])
        test_s |= set(slides[n_tr + n_va:])
    return train_s, val_s, test_s


def make_split(
    df: pd.DataFrame,
    out_dir: Path,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
    group_col: str | None = None,
    label_col: str = "label",
) -> dict:
    """Split a dataframe and save train/val/test CSVs.

    If ``group_col`` is provided, the split is performed over unique groups
    (slides) so patches from one slide never span two subsets. Otherwise it falls
    back to a per-row stratified split.

    Returns a summary dict with per-split counts.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if group_col and group_col in df.columns and df[group_col].nunique() >= 3:
        tr_g, va_g, te_g = _group_split(df, group_col, label_col, train, val, seed)
        train_df = df[df[group_col].isin(tr_g)]
        val_df = df[df[group_col].isin(va_g)]
        test_df = df[df[group_col].isin(te_g)]
        n_groups = {"train": len(tr_g), "val": len(va_g), "test": len(te_g)}
    else:
        strat = df[label_col] if (stratify and label_col in df.columns) else None
        train_df, temp_df = train_test_split(
            df, train_size=train, stratify=strat, random_state=seed)
        rel_val = val / (val + test)
        strat_temp = temp_df[label_col] if (stratify and label_col in temp_df.columns) else None
        val_df, test_df = train_test_split(
            temp_df, train_size=rel_val, stratify=strat_temp, random_state=seed)
        n_groups = None

    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = out_dir / f"{name}.csv"
        try:
            frame.to_csv(path, index=False)
        except PermissionError as exc:
            raise PermissionError(
                f"Could not write '{path}'. The file is open in another program "
                f"(usually Excel). Please close it and run the pipeline again."
            ) from exc

    summary = {
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "class_distribution": df[label_col].value_counts().to_dict()
        if label_col in df.columns else {},
        "split_level": "slide" if n_groups else "patch",
    }
    if n_groups:
        summary["slides"] = n_groups
    return summary
