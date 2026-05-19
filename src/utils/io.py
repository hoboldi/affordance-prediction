from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path, *, indent: int = 2) -> None:
    Path(path).write_text(json.dumps(data, indent=indent))


def find_mask_file(masks_dir: Path, stem: str) -> Path | None:
    for ext in MASK_EXTENSIONS:
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_rgb_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_binary_mask(path: str | Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., -1]
    return (arr > 0).astype(bool)


def full_image_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape[:2], dtype=bool)


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
