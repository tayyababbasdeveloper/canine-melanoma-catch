"""Reproducibility helpers.

A single place to seed Python, NumPy and (when available) PyTorch so that
training runs are repeatable — required for the cross-validation and
architecture-comparison experiments described in the proposal (section 3.1.4).
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all relevant RNGs. Returns the seed for convenience/logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional at import time (only needed from Week 3 onward)
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
