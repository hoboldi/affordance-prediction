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


def _load_vertex_positions(recon_dir: Path) -> torch.Tensor | None:
    """Per-vertex xyz from ``mesh.glb``, centered + unit-sphere normalized, V-aligned to
    the other reconstruction artifacts. Cached as ``vertex_positions.pt`` after first load."""
    cache = recon_dir / "vertex_positions.pt"
    if cache.is_file():
        return torch.load(cache, map_location="cpu", weights_only=True)
    mesh_path = recon_dir / "mesh.glb"
    if not mesh_path.is_file():
        return None
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    xyz = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-6)  # unit sphere — preserves aspect ratio
    torch.save(xyz, cache)
    return xyz


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
    vertex_pseudolabel_path: Path | None = None
    vertex_semantics_path: Path | None = None
    sam3d_reconstruction_dir: Path | None = None
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
        "vertex_pseudolabel_path",
        "vertex_semantics_path",
        "sam3d_reconstruction_dir",
        "split",
    }
    extras = {k: v for k, v in raw.items() if k not in known}

    _CONTAINER_DATA_ROOT = Path("/workspace/data")

    def p(key: str) -> Path | None:
        if key not in raw or raw[key] is None or raw[key] == "":
            return None
        path = Path(str(raw[key]))
        if path.is_absolute():
            try:
                rel = path.relative_to(_CONTAINER_DATA_ROOT)
                return data_root / rel
            except ValueError:
                return path
        return data_root / path

    return ManifestRow(
        sample_id=str(raw["sample_id"]),
        verb=str(raw.get("verb", "grasp")),
        mesh_path=p("mesh_path"),
        splat_path=p("splat_path"),
        reference_rgb_path=p("reference_rgb_path"),
        mask_path=p("mask_path"),
        vertex_affordance_path=p("vertex_affordance_path"),
        vertex_pseudolabel_path=p("vertex_pseudolabel_path"),
        vertex_semantics_path=p("vertex_semantics_path"),
        sam3d_reconstruction_dir=p("sam3d_reconstruction_dir"),
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


def _load_vertex_semantics_bundle(path: Path) -> dict[str, torch.Tensor]:
    """Load ``vertex_semantic.pt``-style dict: ``features`` (V, D), ``visible_in_any_view`` (V,)."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "features" not in data:
        raise ValueError(
            f"Expected dict with 'features' tensor from projection cache, got keys {set(data) if isinstance(data, dict) else type(data)}: {path}"
        )
    feats = data["features"].float()
    vis = data.get("visible_in_any_view")
    if vis is None:
        mask = torch.ones(feats.shape[0], dtype=torch.bool)
    else:
        mask = torch.from_numpy(np.asarray(vis, dtype=bool))
    return {"vertex_features": feats, "vertex_visible_mask": mask}


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
    Heavy fields (mesh, labels, vertex semantics, SAM3D latent) load lazily depending on constructor flags.
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
        load_vertex_semantics_eager: bool | None = None,
        load_vertex_dino: bool = False,
        dino_filename: str = "vertex_dino.pt",
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
        if load_vertex_semantics_eager is None:
            load_vertex_semantics_eager = ds.get("load_vertex_semantics_eager", True)
        if isinstance(load_vertex_semantics_eager, str):
            load_vertex_semantics_eager = load_vertex_semantics_eager.lower() in ("1", "true", "yes")

        self._load_mesh_eager = load_mesh_eager
        self._load_vertex_labels_eager = load_vertex_labels_eager
        self._load_vertex_semantics_eager = bool(load_vertex_semantics_eager)
        self._load_vertex_dino = bool(load_vertex_dino)
        self._dino_filename = str(dino_filename)
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

    @property
    def rows(self) -> list:
        return self._rows

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

        # Affordance supervision: GEAL pseudolabels take priority, then any precomputed GT
        # (.npy / .pt of shape (V,)). Labels are already in mesh-vertex order — no alignment.
        label_path = row.vertex_pseudolabel_path or row.vertex_affordance_path
        if label_path is not None:
            out["vertex_affordance_path"] = label_path
            out["vertex_affordance"] = (
                _load_vertex_affordance(label_path) if self._load_vertex_labels_eager else None
            )
        else:
            out["vertex_affordance_path"] = None
            out["vertex_affordance"] = None

        if row.vertex_semantics_path is not None:
            out["vertex_semantics_path"] = row.vertex_semantics_path
            if self._load_vertex_semantics_eager:
                sem = _load_vertex_semantics_bundle(row.vertex_semantics_path)
                out["vertex_features"] = sem["vertex_features"]
                out["vertex_visible_mask"] = sem["vertex_visible_mask"]
            else:
                out["vertex_features"] = None
                out["vertex_visible_mask"] = None
        else:
            out["vertex_semantics_path"] = None
            out["vertex_features"] = None
            out["vertex_visible_mask"] = None

        if row.sam3d_reconstruction_dir is not None:
            out["sam3d_reconstruction_dir"] = row.sam3d_reconstruction_dir
            out["slat_vertex_features"] = torch.load(
                row.sam3d_reconstruction_dir / "slat_vertex_features.pt",
                map_location="cpu",
                weights_only=True,
            )
            _global_latent_path = row.sam3d_reconstruction_dir / "global_latent.pt"
            if _global_latent_path.is_file():
                _blob = torch.load(_global_latent_path, map_location="cpu", weights_only=True)
                out["global_latent"] = _blob["global_latent"] if isinstance(_blob, dict) else _blob
            else:
                out["global_latent"] = None
            for _key in ("dino_cls", "ss_dino_cls", "vertex_normals"):
                _pt = row.sam3d_reconstruction_dir / f"{_key}.pt"
                out[_key] = (
                    torch.load(_pt, map_location="cpu", weights_only=True)
                    if _pt.is_file()
                    else None
                )
            out["vertex_positions"] = _load_vertex_positions(row.sam3d_reconstruction_dir)

            if self._load_vertex_dino:
                _dino_pt = row.sam3d_reconstruction_dir / self._dino_filename
                if _dino_pt.is_file():
                    _d = torch.load(_dino_pt, map_location="cpu", weights_only=True)
                    out["dino_vertex_features"] = (_d["features"].float() if isinstance(_d, dict) and "features" in _d else _d.float())
                else:
                    out["dino_vertex_features"] = None
            else:
                out["dino_vertex_features"] = None

            # Canonicalize orientation so per-vertex positions/normals share an upright frame across
            # objects (GEAL labelled at this same orientation). R from generate_geal_pseudolabels;
            # canonical coords = X @ R.T (matches the points fed to GEAL).
            _canon = row.sam3d_reconstruction_dir / "canonical_rotation.pt"
            if _canon.is_file():
                _R = torch.load(_canon, map_location="cpu", weights_only=True).float()
                if out.get("vertex_positions") is not None:
                    out["vertex_positions"] = out["vertex_positions"].float() @ _R.T
                if out.get("vertex_normals") is not None:
                    out["vertex_normals"] = out["vertex_normals"].float() @ _R.T
        else:
            out["sam3d_reconstruction_dir"] = None
            out["slat_vertex_features"] = None
            out["global_latent"] = None
            out["dino_cls"] = None
            out["ss_dino_cls"] = None
            out["vertex_normals"] = None
            out["vertex_positions"] = None

        return out
