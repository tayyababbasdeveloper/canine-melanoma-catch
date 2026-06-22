"""PyTorch Dataset for tumour segmentation patches.

Reads a manifest CSV with columns ['image_path', 'mask_path', 'slide_id', 'mag']
(produced by scripts/prepare_segmentation_data.py) and returns
(image_tensor, mask_tensor) pairs ready for a U-Net.

Masks are binarised to {0, 1} and returned with a channel dimension (1, H, W)
to match the single-logit output of the segmentation head.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class SegmentationPatchDataset(Dataset):
    def __init__(self, manifest, transform=None):
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        else:
            self.df = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not read image: {row['image_path']}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read mask: {row['mask_path']}")
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            # albumentations ToTensorV2 leaves the mask as (H, W); add channel dim
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(np.asarray(mask))
            mask = mask.unsqueeze(0).float()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

    def positive_fraction(self) -> float:
        """Mean foreground (tumour) pixel fraction across a sample of masks.

        Used to set the BCE ``pos_weight`` that counters class imbalance
        (proposal risk table: class imbalance — High likelihood).
        """
        sample = self.df["mask_path"].head(200)
        fracs = []
        for p in sample:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                fracs.append((m > 127).mean())
        return float(np.mean(fracs)) if fracs else 0.0
