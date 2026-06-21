from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from reconstruction.sam3d_wrapper import load_global_latent, save_global_latent
from utils.config import project_root
from utils.io import save_json


@dataclass(frozen=True)
class ReconstructionArtifacts:
    """Standard on-disk filenames for one reconstruction sample."""

    shape_latent: str = "shape_latent.pt"
    slat_feats: str = "slat_feats.pt"
    slat_coords: str = "slat_coords.pt"
    slat_vertex_features: str = "slat_vertex_features.pt"
    global_latent: str = "global_latent.pt"
    dino_cls: str = "dino_cls.pt"
    dino_patches: str = "dino_patches.pt"
    ss_dino_cls: str = "ss_dino_cls.pt"
    ss_dino_patches: str = "ss_dino_patches.pt"
    vertex_normals: str = "vertex_normals.pt"
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
        "slat_vertex_features": out_dir / artifacts.slat_vertex_features,
        "global_latent": out_dir / artifacts.global_latent,
        "dino_cls": out_dir / artifacts.dino_cls,
        "dino_patches": out_dir / artifacts.dino_patches,
        "ss_dino_cls": out_dir / artifacts.ss_dino_cls,
        "ss_dino_patches": out_dir / artifacts.ss_dino_patches,
        "vertex_normals": out_dir / artifacts.vertex_normals,
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


def _dataset_path_tail_key(path: Path) -> str | None:
    """
    Return a lowercase ``Seen/...`` or ``UnSeen/...`` tail so we can match Docker vs host paths.

    Falls back to ``None`` when the path does not contain those markers (e.g. tiny examples).
    """
    parts = tuple(x.lower() for x in path.parts)
    for marker in ("seen", "unseen"):
        if marker in parts:
            i = parts.index(marker)
            return "/".join(path.parts[i:]).lower()
    return None


def _paths_refer_to_same_splat(recorded: str, want: Path) -> bool:
    """Whether ``meta_prerender.json`` ``splat_path`` refers to the same on-disk Gaussian as ``want``."""
    want_p = Path(want).expanduser().resolve()
    rec_p = Path(recorded).expanduser()
    try:
        rp = rec_p.resolve()
        if rp.exists() and want_p.exists() and rp.samefile(want_p):
            return True
        if rp == want_p:
            return True
    except OSError:
        pass
    tail_w = _dataset_path_tail_key(want_p)
    try:
        tail_r = _dataset_path_tail_key(rec_p)
    except OSError:
        tail_r = None
    if tail_w and tail_r and tail_w == tail_r:
        return rec_p.name.lower() == want_p.name.lower()
    return False


def prerender_meta_matches_splat(meta_prerender_json: Path | str, splat_ply: str | Path) -> bool:
    """
    Return whether ``sam3d_dataset/meta_prerender.json`` lists ``splat_path`` consistent with ``splat_ply``.

    Used to skip redundant SAM3D runs and to validate cached exports.
    """
    meta_path = Path(meta_prerender_json).expanduser()
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rec = meta.get("splat_path")
    if not isinstance(rec, str) or not rec.strip():
        return False
    return _paths_refer_to_same_splat(rec, splat_ply)


def find_sam3d_reconstruction_mesh_for_splat(
    splat_ply: str | Path,
    *,
    search_under: Path | str | None = None,
) -> Path | None:
    """
    Find ``<run_dir>/reconstruction/mesh.glb`` produced by a SAM3D reconstruction run
    for **this** splat file.

    Each run writes ``<run_dir>/sam3d_dataset/meta_prerender.json`` with a ``splat_path`` field.
    If several runs match (re-runs), returns the **newest** ``mesh.glb`` by mtime.

    Returns ``None`` when no matching run exists under ``<search_under>/exports`` (not an error).
    """
    root = Path(search_under).expanduser().resolve() if search_under is not None else project_root()
    exports = root / "exports"
    if not exports.is_dir():
        return None
    want = Path(splat_ply).expanduser().resolve()
    candidates: list[Path] = []
    for meta_path in exports.glob("**/sam3d_dataset/meta_prerender.json"):
        mesh = meta_path.parent.parent / "reconstruction" / "mesh.glb"
        if not mesh.is_file():
            continue
        if not prerender_meta_matches_splat(meta_path, want):
            continue
        candidates.append(mesh)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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


def _apply_world_rotation_to_trimesh_export(obj: Any, R: np.ndarray) -> None:
    """Apply ``R @ v`` to every ``Trimesh`` vertex (in-place). ``obj`` is a ``Trimesh`` or ``Scene``."""
    if isinstance(obj, trimesh.Trimesh):
        v = np.asarray(obj.vertices, dtype=np.float64)
        obj.vertices = (R @ v.T).T.astype(np.float32, copy=False)
        return
    if isinstance(obj, trimesh.Scene):
        for g in obj.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                _apply_world_rotation_to_trimesh_export(g, R)
        return
    raise TypeError(f"expected trimesh.Trimesh or Scene for mesh export; got {type(obj).__name__}")


def _apply_world_rotation_to_sam3d_gaussian(gs: Any, R: np.ndarray) -> None:
    """
    Rotate decoded 3DGS means and orientations into the same frame as ``mesh.glb``.

    Uses ``pytorch3d`` when available (SAM3D Docker); otherwise is a no-op so the mesh-only
    correction still runs in lightweight environments.
    """
    try:
        from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
    except ImportError:
        return

    R_t = torch.as_tensor(R, dtype=torch.float32, device=gs.get_xyz.device)
    xyz = gs.get_xyz
    gs.from_xyz((R_t @ xyz.T).T)
    q = gs.get_rotation
    M = quaternion_to_matrix(q)
    M_new = torch.einsum("ij,njk->nik", R_t, M)
    gs.from_rotation(matrix_to_quaternion(M_new))


def apply_vertex_world_rotation_to_sam3d_result(result: Any, R: np.ndarray | None) -> None:
    """
    Mutate ``result`` mesh / gaussian assets in-place before :func:`save_reconstruction`.

    ``R`` is an orthogonal 3×3 acting on **column** vertex vectors ``p' = R @ p``, matching
    :func:`rendering.camera_sampling.rotation_matrix_from_axis_angle_deg`.
    """
    if R is None:
        return
    R = np.asarray(R, dtype=np.float64)
    if result.mesh_scene is not None:
        _apply_world_rotation_to_trimesh_export(result.mesh_scene, R)
    if result.gaussian_splat is not None:
        _apply_world_rotation_to_sam3d_gaussian(result.gaussian_splat, R)


def _extract_mesh_vertices(mesh_scene: Any) -> np.ndarray | None:
    """Return (V, 3) float32 vertex positions from a trimesh Trimesh or Scene, or None."""
    if mesh_scene is None:
        return None
    if isinstance(mesh_scene, trimesh.Trimesh):
        return np.array(mesh_scene.vertices, dtype=np.float32)
    if isinstance(mesh_scene, trimesh.Scene):
        parts = [
            np.array(g.vertices, dtype=np.float32)
            for g in mesh_scene.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        return np.concatenate(parts, axis=0) if parts else None
    return None


def _extract_mesh_vertex_normals(mesh_scene: Any) -> np.ndarray | None:
    """Return (V, 3) float32 vertex normals from a trimesh Trimesh or Scene, or None."""
    if mesh_scene is None:
        return None
    if isinstance(mesh_scene, trimesh.Trimesh):
        return np.array(mesh_scene.vertex_normals, dtype=np.float32)
    if isinstance(mesh_scene, trimesh.Scene):
        parts = [
            np.array(g.vertex_normals, dtype=np.float32)
            for g in mesh_scene.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        return np.concatenate(parts, axis=0) if parts else None
    return None


def save_reconstruction(
    result: Any,
    out_dir: Path,
    *,
    stem: str,
    seed: int,
    image_path: Path | None = None,
    mask_path: Path | None = None,
    vertex_world_rotation: np.ndarray | None = None,
) -> dict[str, Path]:
    """
    Persist a :class:`~reconstruction.sam3d_wrapper.ReconstructionResult` to disk.

    Writes ``global_latent.pt`` (mean-pooled SLAT, same format as :func:`~reconstruction.sam3d_wrapper.save_global_latent`)
    next to ``slat_vertex_features.pt`` under ``out_dir``.

    When ``vertex_world_rotation`` is a 3×3 matrix, it is applied to the decoded mesh (and SAM3D
    Gaussian means + rotations when ``pytorch3d`` is importable) **before** writing ``mesh.glb`` /
    ``gaussian.ply``. Used to undo a fixed orbit-ring tilt of cameras vs the normalized splat frame.

    Returns the artifact paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = reconstruction_paths(out_dir)

    torch.save(result.shape_latent, paths["shape_latent"])
    torch.save(result.slat_feats, paths["slat_feats"])
    torch.save(result.slat_coords, paths["slat_coords"])

    # Compute per-vertex SLAT features while mesh and voxels share the same (pre-rotation) frame.
    verts_np = _extract_mesh_vertices(result.mesh_scene)
    if verts_np is not None:
        slat_vertex = slat_feats_to_vertex_features(
            result.slat_feats,
            result.slat_coords.float(),
            torch.from_numpy(verts_np),
        )
        torch.save(slat_vertex, paths["slat_vertex_features"])

    gl = getattr(result, "global_latent", None)
    if gl is not None:
        save_global_latent(gl, paths["global_latent"])

    dino_cls = getattr(result, "dino_cls", None)
    if dino_cls is not None:
        torch.save(dino_cls, paths["dino_cls"])

    dino_patches = getattr(result, "dino_patches", None)
    if dino_patches is not None:
        torch.save(dino_patches, paths["dino_patches"])

    ss_dino_cls = getattr(result, "ss_dino_cls", None)
    if ss_dino_cls is not None:
        torch.save(ss_dino_cls, paths["ss_dino_cls"])

    ss_dino_patches = getattr(result, "ss_dino_patches", None)
    if ss_dino_patches is not None:
        torch.save(ss_dino_patches, paths["ss_dino_patches"])

    vertex_normals_np = _extract_mesh_vertex_normals(result.mesh_scene)
    if vertex_normals_np is not None:
        torch.save(torch.from_numpy(vertex_normals_np), paths["vertex_normals"])

    apply_vertex_world_rotation_to_sam3d_result(result, vertex_world_rotation)

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
        "slat_vertex_features_shape": list(slat_vertex.shape) if verts_np is not None else None,
        "global_latent_shape": list(gl.shape) if gl is not None else None,
        "dino_cls_shape": list(dino_cls.shape) if dino_cls is not None else None,
        "dino_patches_shape": list(dino_patches.shape) if dino_patches is not None else None,
        "ss_dino_cls_shape": list(ss_dino_cls.shape) if ss_dino_cls is not None else None,
        "ss_dino_patches_shape": list(ss_dino_patches.shape) if ss_dino_patches is not None else None,
        "vertex_normals_shape": list(vertex_normals_np.shape) if vertex_normals_np is not None else None,
        "mesh_path": str(paths["mesh"]) if paths["mesh"].exists() else None,
        "vertex_world_rotation": vertex_world_rotation.astype(float).tolist()
        if vertex_world_rotation is not None
        else None,
    }
    save_json(meta, paths["meta"])
    return paths


def load_latents(out_dir: Path) -> dict[str, torch.Tensor]:
    """Load cached SAM3D latents from a reconstruction directory."""
    paths = reconstruction_paths(out_dir)
    out: dict[str, torch.Tensor] = {
        "shape_latent": torch.load(paths["shape_latent"], map_location="cpu", weights_only=True),
        "slat_feats": torch.load(paths["slat_feats"], map_location="cpu", weights_only=True),
        "slat_coords": torch.load(paths["slat_coords"], map_location="cpu", weights_only=True),
    }
    if paths["slat_vertex_features"].is_file():
        out["slat_vertex_features"] = torch.load(
            paths["slat_vertex_features"], map_location="cpu", weights_only=True
        )
    if paths["global_latent"].is_file():
        out["global_latent"] = load_global_latent(paths["global_latent"])
    if paths["dino_cls"].is_file():
        out["dino_cls"] = torch.load(paths["dino_cls"], map_location="cpu", weights_only=True)
    if paths["dino_patches"].is_file():
        out["dino_patches"] = torch.load(paths["dino_patches"], map_location="cpu", weights_only=True)
    if paths["ss_dino_cls"].is_file():
        out["ss_dino_cls"] = torch.load(paths["ss_dino_cls"], map_location="cpu", weights_only=True)
    if paths["ss_dino_patches"].is_file():
        out["ss_dino_patches"] = torch.load(paths["ss_dino_patches"], map_location="cpu", weights_only=True)
    if paths["vertex_normals"].is_file():
        out["vertex_normals"] = torch.load(paths["vertex_normals"], map_location="cpu", weights_only=True)
    return out


def slat_feats_to_vertex_features(
    slat_feats: torch.Tensor,
    slat_coords: torch.Tensor,
    vertex_positions: torch.Tensor,
) -> torch.Tensor:
    """Assign each mesh vertex the feature of its nearest SLAT voxel.

    Args:
        slat_feats:        (N, C) per-voxel features from SAM3D
        slat_coords:       (N, 3) voxel positions, same coordinate frame as vertex_positions
        vertex_positions:  (V, 3) mesh vertex positions

    Returns:
        (V, C) per-vertex features
    """
    from scipy.spatial import cKDTree

    coords_np = slat_coords.float().cpu().numpy()
    verts_np = vertex_positions.float().cpu().numpy()
    _, nn_idx = cKDTree(coords_np).query(verts_np, k=1, workers=-1)
    return slat_feats[torch.from_numpy(nn_idx).long()]
