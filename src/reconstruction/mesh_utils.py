from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.io import save_json


@dataclass(frozen=True)
class ReconstructionArtifacts:
    """Standard on-disk filenames for one reconstruction sample."""

    shape_latent: str = "shape_latent.pt"
    slat_feats: str = "slat_feats.pt"
    slat_coords: str = "slat_coords.pt"
    gaussian: str = "gaussian.ply"
    mesh: str = "mesh.glb"
    meta: str = "meta.json"


def reconstruction_paths(out_dir: Path) -> dict[str, Path]:
    """Return absolute paths for all standard reconstruction artifacts."""
    artifacts = ReconstructionArtifacts()
    return {
        "out_dir": out_dir,
        "shape_latent": out_dir / artifacts.shape_latent,
        "slat_feats": out_dir / artifacts.slat_feats,
        "slat_coords": out_dir / artifacts.slat_coords,
        "gaussian": out_dir / artifacts.gaussian,
        "mesh": out_dir / artifacts.mesh,
        "meta": out_dir / artifacts.meta,
    }


def ensure_decode_formats(decode_formats: list[str]) -> list[str]:
    """
    SAM3D mesh decoding requires the gaussian branch in ``postprocess_slat_output``.
    """
    formats = list(decode_formats)
    if "mesh" in formats and "gaussian" not in formats:
        formats = ["gaussian", *formats]
    return formats


def save_reconstruction(
    result: Any,
    out_dir: Path,
    *,
    stem: str,
    seed: int,
    image_path: Path | None = None,
    mask_path: Path | None = None,
) -> dict[str, Path]:
    """
    Persist a :class:`~reconstruction.sam3d_wrapper.ReconstructionResult` to disk.

    Returns the artifact paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = reconstruction_paths(out_dir)

    torch.save(result.shape_latent, paths["shape_latent"])
    torch.save(result.slat_feats, paths["slat_feats"])
    torch.save(result.slat_coords, paths["slat_coords"])

    if result.gaussian_splat is not None:
        result.gaussian_splat.save_ply(str(paths["gaussian"]))

    if result.mesh_scene is not None:
        result.mesh_scene.export(str(paths["mesh"]))

    meta = {
        "stem": stem,
        "seed": seed,
        "image_path": str(image_path) if image_path else None,
        "mask_path": str(mask_path) if mask_path else None,
        "shape_latent_shape": list(result.shape_latent.shape),
        "slat_voxel_count": int(result.slat_coords.shape[0]),
        "slat_feats_shape": list(result.slat_feats.shape),
        "mesh_path": str(paths["mesh"]) if paths["mesh"].exists() else None,
    }
    save_json(meta, paths["meta"])
    return paths


def load_latents(out_dir: Path) -> dict[str, torch.Tensor]:
    """Load cached SAM3D latents from a reconstruction directory."""
    paths = reconstruction_paths(out_dir)
    return {
        "shape_latent": torch.load(paths["shape_latent"], map_location="cpu", weights_only=True),
        "slat_feats": torch.load(paths["slat_feats"], map_location="cpu", weights_only=True),
        "slat_coords": torch.load(paths["slat_coords"], map_location="cpu", weights_only=True),
    }
