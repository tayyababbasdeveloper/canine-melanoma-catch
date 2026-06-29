"""PyTorch Dataset for tumour-subtype classification patches.

Reads a manifest CSV with columns ['image_path', 'label', 'label_idx', 'slide_id']
and returns (image_tensor, label_idx). Image paths are portable relative paths
resolved against the project root (same approach as the segmentation dataset).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else _PROJECT_ROOT / path)


class ClassificationPatchDataset(Dataset):
    def __init__(self, manifest, transform=None):
        self.df = (pd.read_csv(manifest) if isinstance(manifest, (str, Path))
                   else manifest.reset_index(drop=True))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        bgr = cv2.imread(_resolve(row["image_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not read image: {row['image_path']}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        label = int(row["label_idx"])

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image, label

    def class_counts(self) -> dict:
        return self.df["label"].value_counts().to_dict()
