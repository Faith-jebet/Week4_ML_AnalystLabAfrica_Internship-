"""
Central configuration loader.

Week 6: wires config.yaml into the pipeline (flagged as an open item in the
Week 5 report). Modules import get_config() instead of hard-coding paths
and thresholds directly, so there is one place to change settings.
"""

import logging
from pathlib import Path

import yaml

_CONFIG_CACHE = None


def get_config(path: str = "config.yaml") -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at '{path}' — run from the repo root.")
    with open(config_path) as f:
        _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def get_logger(name: str) -> logging.Logger:
    """Shared logger setup so every pipeline stage logs consistently (Week 6: was print()-only)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                                 datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
