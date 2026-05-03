"""TOML configuration loading helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load a TOML configuration file into a plain dictionary."""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)
