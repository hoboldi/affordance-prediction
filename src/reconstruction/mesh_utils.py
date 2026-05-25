from __future__ import annotations

import copy
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


def sam3d_run_reconstruction_paths(run_dir: str | Path) -> dict[str, Path]:
    """
    Artifact paths for a pipeline run that wrote SAM3D under ``<run_dir>/reconstruction/``.

    ``run_dir`` is the same directory as ``RUN_DIR`` in ``notebooks/10_sam3d_from_gsplat.ipynb``
    (the parent of ``reconstruction/``, not the ``reconstruction`` folder itself).
    """
    return reconstruction_paths(Path(run_dir).expanduser().resolve() / "reconstruction")


def cfg_with_sam3d_reconstruction(
    cfg: dict[str, Any],
    run_dir: str | Path,
    *,
    link_gaussian_splat: bool = True,
) -> dict[str, Any]:
    """
    Deep-copy ``cfg`` and point ``rendering.mesh_path`` at SAM3D's ``mesh.glb``.

    Optionally sets ``rendering.splat_path`` to ``gaussian.ply`` when present and bumps
    ``rendering.backend`` from ``mesh`` → ``gaussian`` so splat RGB is used with the SAM3D mesh
    correspondences (see ``rendering.renderer.render_mesh_views``).
    """
    out = copy.deepcopy(cfg)
    paths = sam3d_run_reconstruction_paths(run_dir)
    mesh = paths["mesh"]
    if not mesh.is_file():
        raise FileNotFoundError(
            f"SAM3D mesh not found at {mesh}. Use the same RUN_DIR as notebook 10 "
            "(parent of reconstruction/ containing mesh.glb)."
        )
    rendering = out.setdefault("rendering", {})
    rendering["mesh_path"] = str(mesh.resolve())
    gaussian = paths["gaussian"]
    if link_gaussian_splat and gaussian.is_file():
        rendering["splat_path"] = str(gaussian.resolve())
        if rendering.get("backend") == "mesh":
            rendering["backend"] = "gaussian"
    else:
        rendering["splat_path"] = None
    if rendering.get("backend") in ("gaussian", "both") and not rendering.get("splat_path"):
        rendering["backend"] = "mesh"
    return out


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
