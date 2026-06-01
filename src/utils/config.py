"""Configuration loader.

Loads the YAML config and resolves all paths relative to the project root so the
pipeline can be run from any working directory.
"""
from __future__ import annotations

from pathlib import Path
import yaml

# Project root = two levels up from this file (src/utils/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.yaml and absolutise all paths under the 'paths' section."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve every path relative to the project root
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = (PROJECT_ROOT / value).resolve()

    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Create all output directories declared in the config if missing."""
    for key in ("interim_dir", "processed_dir", "patches_dir",
                "figures_dir", "logs_dir"):
        path = cfg["paths"].get(key)
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
