"""Stratified train / validation / test split.

Splits patches into 70/15/15 subsets while preserving the class distribution
(proposal section 3.1.1: stratified sampling). Writes train.csv / val.csv /
test.csv listing each patch path and its class label.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(
    df: pd.DataFrame,
    out_dir: Path,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> dict:
    """Split a dataframe with columns ['patch_path', 'label'] and save CSVs.

    Returns a summary dict with per-split counts.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strat = df["label"] if stratify else None
    train_df, temp_df = train_test_split(
        df, train_size=train, stratify=strat, random_state=seed
    )

    rel_val = val / (val + test)
    strat_temp = temp_df["label"] if stratify else None
    val_df, test_df = train_test_split(
        temp_df, train_size=rel_val, stratify=strat_temp, random_state=seed
    )

    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = out_dir / f"{name}.csv"
        try:
            frame.to_csv(path, index=False)
        except PermissionError as exc:
            raise PermissionError(
                f"Could not write '{path}'. The file is open in another program "
                f"(usually Excel). Please close it and run the pipeline again."
            ) from exc

    return {
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "class_distribution": df["label"].value_counts().to_dict(),
    }
