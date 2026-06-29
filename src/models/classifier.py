"""Tumour-subtype classifiers (proposal section 3.1.3).

Two ImageNet-pretrained architectures, fine-tuned with transfer learning:
  * ResNet-50        (He et al., 2016)
  * EfficientNet-B3  (Tan & Le, 2019)

A progressive-unfreezing helper lets training start with only the classification
head trainable and then gradually unfreeze deeper layers — the strategy named in
the proposal to guard against catastrophic forgetting on a small dataset.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_classifier(arch: str = "resnet50", num_classes: int = 3,
                     pretrained: bool = True, logger=None) -> nn.Module:
    """Build an ImageNet-pretrained classifier with a fresh head.

    Falls back to randomly-initialised weights if the pretrained weights cannot
    be downloaded (offline environments).
    """
    def _weights(enum):
        if not pretrained:
            return None
        try:
            return enum.DEFAULT
        except Exception as exc:  # pragma: no cover
            msg = f"Pretrained weights unavailable ({exc}); using random init."
            (logger.warning if logger else print)(msg)
            return None

    if arch == "resnet50":
        model = models.resnet50(weights=_weights(models.ResNet50_Weights))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=_weights(models.EfficientNet_B3_Weights))
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown classifier arch: {arch}")
    return model


def set_trainable_stage(model: nn.Module, arch: str, stage: int) -> None:
    """Progressive unfreezing.

    stage 0 -> only the classification head trainable;
    stage 1 -> head + last block; stage 2 -> entire network.
    """
    for p in model.parameters():
        p.requires_grad = False

    if arch == "resnet50":
        head = [model.fc]
        blocks = [model.layer4, model.layer3]
    else:  # efficientnet_b3
        head = [model.classifier]
        feats = model.features
        blocks = [feats[-1], feats[-2]]

    groups = head + blocks[:stage]
    for g in groups:
        for p in g.parameters():
            p.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
