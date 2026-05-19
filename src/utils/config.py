from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = "configs/default.yaml"


def project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge with ``configs/default.yaml``."""
    root = project_root()
    default_path = root / _DEFAULT_CONFIG
    merged: dict[str, Any] = {}
    if default_path.exists():
        with default_path.open() as f:
            merged = yaml.safe_load(f) or {}

    if config_path is not None:
        path = Path(config_path)
        if not path.is_absolute():
            path = root / path
        with path.open() as f:
            override = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, override)

    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a path relative to the project root if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or project_root()) / p
