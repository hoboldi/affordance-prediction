from __future__ import annotations

from pathlib import Path

import torch

from vlm.patch_extractor import PatchFeatures


def save_patch_features(features: PatchFeatures, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "patches": features.patches,
            "grid_h": features.grid_h,
            "grid_w": features.grid_w,
            "feature_dim": features.feature_dim,
        },
        path,
    )


def load_patch_features(path: str | Path) -> PatchFeatures:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return PatchFeatures(
        patches=data["patches"],
        grid_h=int(data["grid_h"]),
        grid_w=int(data["grid_w"]),
        feature_dim=int(data["feature_dim"]),
    )


def save_text_embeddings(embeddings: torch.Tensor, texts: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "texts": texts}, path)


def load_text_embeddings(path: str | Path) -> tuple[torch.Tensor, list[str]]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["embeddings"], list(data["texts"])
