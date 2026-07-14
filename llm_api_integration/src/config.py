"""Loads configs/config.yaml into a plain dict.

Kept as a dict (not a class) on purpose — this project has one shallow
config file, so a dataclass would just be extra ceremony for no benefit.
"""

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the YAML config file and return it as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)
