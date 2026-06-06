"""
Render **true gsplat** multi-view RGB from a 3DGS ``.ply``, then run **SAM3D** on one reference view.

**You cannot use this module outside the `sam3d-pipeline` Docker container** in a supported way:
SAM3D’s upstream stack (torch/CUDA, ``flash-attn``, ``sam-3d-objects``) is only guaranteed there.

SAM3D’s public ``reconstruct`` API is **single-image + mask**; extra rendered views are written for
debugging, future fusion, or feeding ``scripts/generate_sam3d.py`` directly.
"""

from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from reconstruction.mesh_utils import ensure_decode_formats, reconstruction_paths, save_reconstruction
from reconstruction.sam3d_wrapper import (
    ReconstructionResult,
    SAM3DWrapper,
    sam3d_environment,
)
from rendering.camera_sampling import rotation_matrix_from_axis_angle_deg
from rendering.gaussian_gsplat_renderer import render_gaussian_splat_gsplat_views
from rendering.mesh_renderer import MeshRenderConfig, RenderView
from rendering.renderer import build_render_config
from utils.config import load_config, project_root, resolve_path
from utils.io import load_binary_mask, load_rgb_image

_SAM3D_AZIMUTH_DEG: float = 45.0


def _view_stem(index: int) -> str:
    return f"view_{index:03d}"


def _sam3d_render_config(cfg: dict[str, Any], mrc: MeshRenderConfig | None) -> MeshRenderConfig:
    """Single-view +45° render config for SAM3D input when no explicit mrc is given."""
    if mrc is not None:
        return mrc
    base = build_render_config(cfg)
    return replace(base, num_views=1, orbit_azimuth_offsets_deg=(_SAM3D_AZIMUTH_DEG,))


def _vertex_world_rotation_inverse_orbit_ring(
    mrc: MeshRenderConfig | None,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    """
    When cameras use :func:`~rendering.camera_sampling.resolve_spherical_orbit_axis` with a
    non-zero ``orbit_ring_rotation_deg``, splat means stay in the **original** normalized frame
    while SAM3D geometry is easier to interpret in the **tilted** camera convention — apply the
    inverse ring rotation to exports so ``mesh.glb`` / ``gaussian.ply`` match AffordSplat / GT axes.
    """
    eff = mrc if mrc is not None else build_render_config(cfg)
    if abs(eff.orbit_ring_rotation_deg) < 1e-8:
        return None
    return rotation_matrix_from_axis_angle_deg(
        eff.orbit_ring_rotation_axis,
        -float(eff.orbit_ring_rotation_deg),
    )


def export_gsplat_views_for_sam3d(
    ply_path: str | Path,
    dataset_dir: str | Path,
    *,
    mrc: MeshRenderConfig | None = None,
    cfg: dict[str, Any] | None = None,
    max_points: int = 500_000,
    gsplat_seed: int | None = None,
    convert_pyrender_camera_to_gsplat: bool = True,
) -> tuple[list[Path], list[RenderView]]:
    """
    Rasterise the splat with gsplat and write ``dataset_dir/images/{view_xxx}.png`` plus full-foreground
    ``dataset_dir/masks/{view_xxx}.png`` (uint8 0/255) for each view.

    Returns ``(image_paths, render_views)``.
    """
    cfg = cfg if cfg is not None else load_config()
    mrc = _sam3d_render_config(cfg, mrc)

    views = render_gaussian_splat_gsplat_views(
        ply_path,
        mrc,
        max_points=max_points,
        normalize_scene=True,
        seed=gsplat_seed,
        convert_pyrender_camera_to_gsplat=convert_pyrender_camera_to_gsplat,
    )

    root = Path(dataset_dir)
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    meta_views: list[dict[str, Any]] = []
    for i, view in enumerate(views):
        stem = _view_stem(i)
        img_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"
        Image.fromarray(view.rgb).save(img_path)
        h, w = view.rgb.shape[:2]
        mask_u8 = np.ones((h, w), dtype=np.uint8) * 255
        Image.fromarray(mask_u8).save(mask_path)
        image_paths.append(img_path)
        meta_views.append(
            {
                "index": i,
                "stem": stem,
                "rgb": str(img_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
                "camera_to_world": view.camera_pose.astype(float).tolist(),
                "intrinsics": view.intrinsics.astype(float).tolist(),
            }
        )

    (root / "meta_prerender.json").write_text(
        json.dumps({"splat_path": str(Path(ply_path).resolve()), "views": meta_views}, indent=2),
        encoding="utf-8",
    )
    return image_paths, views


def run_sam3d_on_prerendered_view(
    dataset_dir: str | Path,
    recon_out_dir: str | Path,
    wrapper: SAM3DWrapper,
    *,
    reference_view_index: int = 0,
    object_stem: str,
    seed: int | None = 42,
    decode_formats: list[str] | None = None,
    image_path: Path | None = None,
    mask_path: Path | None = None,
    vertex_world_rotation: np.ndarray | None = None,
) -> ReconstructionResult:
    """
    Load ``view_{reference_view_index:03d}.png`` (+ mask) from ``dataset_dir`` and run ``wrapper.reconstruct``.

    ``recon_out_dir`` receives ``save_reconstruction`` outputs (``mesh.glb``, latents, …).

    ``vertex_world_rotation`` — optional 3×3 applied to decoded mesh / Gaussian before export
    (see :func:`_vertex_world_rotation_inverse_orbit_ring`).
    """
    root = Path(dataset_dir)
    images_dir = root / "images"
    masks_dir = root / "masks"
    stem = _view_stem(reference_view_index)
    img_path = image_path or (images_dir / f"{stem}.png")
    m_path = mask_path or (masks_dir / f"{stem}.png")
    if not img_path.is_file():
        raise FileNotFoundError(f"Missing SAM3D RGB: {img_path}")
    if not m_path.is_file():
        raise FileNotFoundError(f"Missing SAM3D mask: {m_path}")

    image = load_rgb_image(img_path)
    mask = load_binary_mask(m_path)
    if decode_formats is None:
        decode_formats = ["gaussian", "mesh"]
    decode_formats = ensure_decode_formats(decode_formats)

    result = wrapper.reconstruct(image, mask, seed=seed, decode_formats=decode_formats)
    save_reconstruction(
        result,
        Path(recon_out_dir),
        stem=object_stem,
        seed=seed if seed is not None else 0,
        image_path=img_path,
        mask_path=m_path,
        vertex_world_rotation=vertex_world_rotation,
    )
    return result


def gsplat_ply_to_sam3d_reconstruction(
    ply_path: str | Path,
    run_dir: str | Path,
    *,
    cfg: dict[str, Any] | None = None,
    wrapper: SAM3DWrapper | None = None,
    object_stem: str | None = None,
    mrc: MeshRenderConfig | None = None,
    max_points: int = 500_000,
    gsplat_seed: int | None = None,
    reference_view_index: int = 0,
    sam3d_seed: int | None = 42,
    decode_formats: list[str] | None = None,
    sam3d_config_path: str | Path | None = None,
    compile_model: bool | None = None,
    convert_pyrender_camera_to_gsplat: bool = True,
) -> dict[str, Any]:
    """
    Full stage: gsplat prerender → SAM3D on one view → artifacts saved under ``run_dir/reconstruction/``.

    Pass a pre-loaded ``wrapper`` to avoid reloading the model across multiple calls (recommended for
    batch processing — loading SAM3D takes ~40 GB and ~30 s each time).  When ``wrapper=None`` a new
    :class:`SAM3DWrapper` is created (and destroyed) for this call only.

    Dense SLAT features (``slat_feats.pt``, ``slat_coords.pt``), per-vertex ``slat_vertex_features.pt``,
    and mean-pooled ``global_latent.pt`` are written by :func:`~reconstruction.mesh_utils.save_reconstruction`.

    Directory layout::

        run_dir/
          sam3d_dataset/images, masks, meta_prerender.json
          reconstruction/{mesh.glb, gaussian.ply, slat_feats.pt, slat_coords.pt, slat_vertex_features.pt, global_latent.pt, …}

    Returns a dict with string keys ``dataset_dir``, ``reconstruction_dir``, ``paths`` (from
    ``reconstruction_paths``), ``result`` (:class:`~reconstruction.sam3d_wrapper.ReconstructionResult`).
    """
    cfg = cfg if cfg is not None else load_config()
    run_root = Path(run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = run_root / "sam3d_dataset"
    recon_dir = run_root / "reconstruction"

    if object_stem is not None:
        stem = object_stem
    else:
        stem = f"{Path(ply_path).stem}_{_view_stem(reference_view_index)}"

    _, _views = export_gsplat_views_for_sam3d(
        ply_path,
        dataset_dir,
        mrc=mrc,
        cfg=cfg,
        max_points=max_points,
        gsplat_seed=gsplat_seed,
        convert_pyrender_camera_to_gsplat=convert_pyrender_camera_to_gsplat,
    )
    # Free gsplat GPU allocations before SAM3D inference to avoid OOM
    del _views
    gc.collect()
    torch.cuda.empty_cache()

    v_rot = _vertex_world_rotation_inverse_orbit_ring(mrc, cfg)

    if wrapper is not None:
        result = run_sam3d_on_prerendered_view(
            dataset_dir,
            recon_dir,
            wrapper,
            reference_view_index=reference_view_index,
            object_stem=stem,
            seed=sam3d_seed,
            decode_formats=decode_formats,
            vertex_world_rotation=v_rot,
        )
    else:
        recon_cfg = cfg.get("reconstruction") or {}
        config_path = sam3d_config_path or recon_cfg.get(
            "sam3d_config", "sam-3d-objects/checkpoints/hf/pipeline.yaml"
        )
        if compile_model is None:
            compile_model = bool(recon_cfg.get("compile", False))
        resolved_config = resolve_path(str(config_path), root=project_root())
        with sam3d_environment(project_root()):
            _wrapper = SAM3DWrapper(resolved_config, compile_model=compile_model)
            result = run_sam3d_on_prerendered_view(
                dataset_dir,
                recon_dir,
                _wrapper,
                reference_view_index=reference_view_index,
                object_stem=stem,
                seed=sam3d_seed,
                decode_formats=decode_formats,
                vertex_world_rotation=v_rot,
            )

    paths = reconstruction_paths(recon_dir)
    out: dict[str, Any] = {
        "dataset_dir": dataset_dir,
        "reconstruction_dir": recon_dir,
        "paths": paths,
        "result": result,
        "object_stem": stem,
    }

    return out
