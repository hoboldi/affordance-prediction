"""
Load training samples from ``<data_root>/manifest.jsonl`` (or a custom manifest path).

``data_root`` defaults to ``paths.data_root`` in config (typically ``data/`` →
``/workspace/data`` in Docker when the repo lives at ``/workspace``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.mesh_loading import MeshData, load_mesh
from utils.config import load_config, resolve_path


@dataclass
class ManifestRow:
    """One line from ``manifest.jsonl`` after path resolution."""

    sample_id: str
    verb: str = "grasp"
    mesh_path: Path | None = None
    splat_path: Path | None = None
    reference_rgb_path: Path | None = None
    mask_path: Path | None = None
    vertex_affordance_path: Path | None = None
    sam3d_global_latent_path: Path | None = None
    split: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _parse_row(raw: dict[str, Any], *, data_root: Path) -> ManifestRow:
    if "sample_id" not in raw:
        raise ValueError(f"manifest row missing sample_id: {raw!r}")

    known = {
        "sample_id",
        "verb",
        "mesh_path",
        "splat_path",
        "reference_rgb_path",
        "mask_path",
        "vertex_affordance_path",
        "sam3d_global_latent_path",
        "split",
    }
    extras = {k: v for k, v in raw.items() if k not in known}

    def p(key: str) -> Path | None:
        if key not in raw or raw[key] is None or raw[key] == "":
            return None
        path = Path(str(raw[key]))
        return path if path.is_absolute() else (data_root / path)

    return ManifestRow(
        sample_id=str(raw["sample_id"]),
        verb=str(raw.get("verb", "grasp")),
        mesh_path=p("mesh_path"),
        splat_path=p("splat_path"),
        reference_rgb_path=p("reference_rgb_path"),
        mask_path=p("mask_path"),
        vertex_affordance_path=p("vertex_affordance_path"),
        sam3d_global_latent_path=p("sam3d_global_latent_path"),
        split=raw.get("split") if raw.get("split") is not None else None,
        extras=extras,
    )


def iter_manifest_rows(
    manifest_path: Path,
    *,
    data_root: Path,
    split: str | None = None,
) -> Iterator[ManifestRow]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {manifest_path}") from e
            if not isinstance(raw, dict):
                raise TypeError(f"Line {lineno} must be a JSON object, got {type(raw)}")
            row = _parse_row(raw, data_root=data_root)
            if split is not None and row.split != split:
                continue
            yield row


def load_manifest_rows(
    manifest_path: Path,
    *,
    data_root: Path,
    split: str | None = None,
) -> list[ManifestRow]:
    return list(iter_manifest_rows(manifest_path, data_root=data_root, split=split))


def _load_vertex_affordance(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if suffix == ".pt":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(blob, torch.Tensor):
            return blob.float()
        if isinstance(blob, dict):
            for key in ("vertex_affordance", "affordance", "labels", "y"):
                if key in blob and isinstance(blob[key], torch.Tensor):
                    return blob[key].float()
        raise ValueError(
            f"Unsupported .pt structure in {path}: expected Tensor or dict with vertex_affordance"
        )
    raise ValueError(f"Unsupported label file extension {suffix} (use .npy or .pt): {path}")


def resolve_data_root(cfg: dict[str, Any] | None = None) -> Path:
    """
    Filesystem root for dataset files (meshes, manifests, caches).

    If the environment variable ``AFFORDANCE_DATA_ROOT`` is set, it wins over config.
    Otherwise ``paths.data_root`` from config is resolved relative to the project root.
    """
    env = os.environ.get("AFFORDANCE_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cfg = cfg if cfg is not None else load_config()
    paths = cfg.get("paths") or {}
    rel = paths.get("data_root", "data")
    return resolve_path(rel)


def resolve_manifest_path(cfg: dict[str, Any] | None, *, data_root: Path | None = None) -> Path:
    cfg = cfg if cfg is not None else load_config()
    root = data_root if data_root is not None else resolve_data_root(cfg)
    ds = cfg.get("dataset") or {}
    name = str(ds.get("manifest_filename", "manifest.jsonl"))
    p = Path(name)
    return p if p.is_absolute() else (root / p)


class DataRootDataset(Dataset[dict[str, Any]]):
    """
    PyTorch ``Dataset`` backed by JSONL manifest under ``data_root``.

    Each ``__getitem__`` returns a dict with string keys suitable for training loops.
    Heavy fields (mesh, labels) load lazily depending on constructor flags.
    """

    def __init__(
        self,
        *,
        data_root: Path | str | None = None,
        manifest_path: Path | str | None = None,
        cfg: dict[str, Any] | None = None,
        split: str | None = None,
        load_mesh_eager: bool | None = None,
        load_vertex_labels_eager: bool | None = None,
        row_filter: Callable[[ManifestRow], bool] | None = None,
    ) -> None:
        cfg = cfg if cfg is not None else load_config()
        self._data_root = Path(data_root).resolve() if data_root is not None else resolve_data_root(cfg)

        ds = cfg.get("dataset") or {}
        if manifest_path is not None:
            mp = Path(manifest_path)
            self._manifest_path = mp if mp.is_absolute() else (self._data_root / mp)
        else:
            self._manifest_path = resolve_manifest_path(cfg, data_root=self._data_root)

        if load_mesh_eager is None:
            load_mesh_eager = bool(ds.get("load_mesh_eager", False))
        if load_vertex_labels_eager is None:
            load_vertex_labels_eager = bool(ds.get("load_vertex_labels_eager", True))

        self._load_mesh_eager = load_mesh_eager
        self._load_vertex_labels_eager = load_vertex_labels_eager
        self._row_filter = row_filter

        rows = load_manifest_rows(self._manifest_path, data_root=self._data_root, split=split)
        if row_filter is not None:
            rows = [r for r in rows if row_filter(r)]
        self._rows = rows

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def __len__(self) -> int:
        return len(self._rows)

    def row(self, index: int) -> ManifestRow:
        return self._rows[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        out: dict[str, Any] = {
            "sample_id": row.sample_id,
            "verb": row.verb,
            "split": row.split,
            "extras": dict(row.extras),
        }

        if row.mesh_path is not None:
            out["mesh_path"] = row.mesh_path
            if self._load_mesh_eager:
                out["mesh"] = load_mesh(row.mesh_path)
            else:
                out["mesh"] = None

        if row.splat_path is not None:
            out["splat_path"] = row.splat_path

        if row.reference_rgb_path is not None:
            out["reference_rgb_path"] = row.reference_rgb_path

        if row.mask_path is not None:
            out["mask_path"] = row.mask_path

        if row.vertex_affordance_path is not None:
            out["vertex_affordance_path"] = row.vertex_affordance_path
            if self._load_vertex_labels_eager:
                out["vertex_affordance"] = _load_vertex_affordance(row.vertex_affordance_path)
            else:
                out["vertex_affordance"] = None
        else:
            out["vertex_affordance_path"] = None
            out["vertex_affordance"] = None

        if row.sam3d_global_latent_path is not None:
            out["sam3d_global_latent_path"] = row.sam3d_global_latent_path
            latent = torch.load(row.sam3d_global_latent_path, map_location="cpu", weights_only=True)
            if isinstance(latent, dict) and "global_latent" in latent:
                out["sam3d_global_latent"] = latent["global_latent"]
            elif isinstance(latent, torch.Tensor):
                out["sam3d_global_latent"] = latent
            else:
                raise ValueError(f"Unexpected SAM3D latent file layout: {row.sam3d_global_latent_path}")

        return out
