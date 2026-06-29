"""Data augmentation for segmentation (proposal section 3.1.1).

Albumentations applies the SAME geometric transform to both the image and its
mask, which is essential for segmentation. Photometric transforms (brightness,
hue/saturation) are applied to the image only and simulate residual H&E stain
and scanner variation that Macenko normalisation does not fully remove.

Augmentation is applied to the training set only; validation/test use a plain
normalise-and-tensor transform so metrics reflect true generalisation.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — the encoders are ImageNet-pretrained (section 3.1.2).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(cfg: dict, patch_size: int = 256) -> A.Compose:
    """Augmentation pipeline for training (image + mask)."""
    a = cfg.get("augmentation", {})
    return A.Compose([
        A.HorizontalFlip(p=a.get("horizontal_flip", 0.5)),
        A.VerticalFlip(p=a.get("vertical_flip", 0.5)),
        A.RandomRotate90(p=a.get("rotate90", 0.5)),
        A.ElasticTransform(alpha=1, sigma=50, p=a.get("elastic_deform", 0.25)),
        A.RandomBrightnessContrast(p=a.get("brightness_contrast", 0.3)),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                             val_shift_limit=10, p=a.get("hue_saturation", 0.3)),
        A.GaussNoise(p=a.get("gauss_noise", 0.2)),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transform(patch_size: int = 256) -> A.Compose:
    """Deterministic transform for validation/test (normalise + to tensor)."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ---- Classification transforms (image only; no mask) ----
def cls_train_transform(cfg: dict) -> A.Compose:
    a = cfg.get("augmentation", {})
    return A.Compose([
        A.HorizontalFlip(p=a.get("horizontal_flip", 0.5)),
        A.VerticalFlip(p=a.get("vertical_flip", 0.5)),
        A.RandomRotate90(p=a.get("rotate90", 0.5)),
        A.RandomBrightnessContrast(p=a.get("brightness_contrast", 0.3)),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                             val_shift_limit=10, p=a.get("hue_saturation", 0.3)),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def cls_eval_transform() -> A.Compose:
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
