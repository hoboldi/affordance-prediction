"""
Render **true gsplat** multi-view RGB from a 3DGS ``.ply``, then run **SAM3D** on one reference view.

SAM3D’s public ``reconstruct`` API is **single-image + mask**; extra rendered views are written for
debugging, future fusion, or feeding ``scripts/generate_sam3d.py`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from reconstruction.mesh_utils import ensure_decode_formats, reconstruction_paths, save_reconstruction
from reconstruction.sam3d_wrapper import (
    ReconstructionResult,
    SAM3DWrapper,
    latent_cache_path,
    save_global_latent,
)
from rendering.gaussian_gsplat_renderer import render_gaussian_splat_gsplat_views
from rendering.mesh_renderer import MeshRenderConfig, RenderView
from rendering.renderer import build_render_config
from utils.config import load_config, project_root, resolve_path
from utils.io import load_binary_mask, load_rgb_image


def _view_stem(index: int) -> str:
    return f"view_{index:03d}"


def export_gsplat_views_for_sam3d(
    ply_path: str | Path,
    dataset_dir: str | Path,
    *,
    mrc: MeshRenderConfig | None = None,
    cfg: dict[str, Any] | None = None,
    max_points: int = 200_000,
    gsplat_seed: int | None = None,
    convert_pyrender_camera_to_gsplat: bool = True,
) -> tuple[list[Path], list[RenderView]]:
    """
    Rasterise the splat with gsplat and write ``dataset_dir/images/{view_xxx}.png`` plus full-foreground
    ``dataset_dir/masks/{view_xxx}.png`` (uint8 0/255) for each view.

    Returns ``(image_paths, render_views)``.
    """
    cfg = cfg if cfg is not None else load_config()
    mrc = mrc if mrc is not None else build_render_config(cfg)

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
) -> ReconstructionResult:
    """
    Load ``view_{reference_view_index:03d}.png`` (+ mask) from ``dataset_dir`` and run ``wrapper.reconstruct``.

    ``recon_out_dir`` receives ``save_reconstruction`` outputs (``mesh.glb``, latents, …).
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
    )
    return result


def gsplat_ply_to_sam3d_reconstruction(
    ply_path: str | Path,
    run_dir: str | Path,
    *,
    cfg: dict[str, Any] | None = None,
    object_stem: str | None = None,
    mrc: MeshRenderConfig | None = None,
    max_points: int = 200_000,
    gsplat_seed: int | None = None,
    reference_view_index: int = 0,
    sam3d_seed: int | None = 42,
    decode_formats: list[str] | None = None,
    sam3d_config_path: str | Path | None = None,
    compile_model: bool | None = None,
    cache_global_latent: bool | None = None,
    convert_pyrender_camera_to_gsplat: bool = True,
) -> dict[str, Any]:
    """
    Full stage: gsplat prerender → SAM3D on one view → optional ``global_latent.pt`` under ``paths.cache_root``.

    Directory layout::

        run_dir/
          sam3d_dataset/images, masks, meta_prerender.json
          reconstruction/{mesh.glb, gaussian.ply, …}

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

    recon_cfg = cfg.get("reconstruction") or {}
    config_path = sam3d_config_path or recon_cfg.get(
        "sam3d_config", "sam-3d-objects/checkpoints/hf/pipeline.yaml"
    )
    if compile_model is None:
        compile_model = bool(recon_cfg.get("compile", False))
    resolved_config = resolve_path(str(config_path), root=project_root())
    wrapper = SAM3DWrapper(resolved_config, compile_model=compile_model)

    result = run_sam3d_on_prerendered_view(
        dataset_dir,
        recon_dir,
        wrapper,
        reference_view_index=reference_view_index,
        object_stem=stem,
        seed=sam3d_seed,
        decode_formats=decode_formats,
    )

    paths = reconstruction_paths(recon_dir)
    out: dict[str, Any] = {
        "dataset_dir": dataset_dir,
        "reconstruction_dir": recon_dir,
        "paths": paths,
        "result": result,
        "object_stem": stem,
    }

    if cache_global_latent is None:
        cache_global_latent = bool(recon_cfg.get("cache_latents", True))
    if cache_global_latent:
        paths_cfg = cfg.get("paths") or {}
        cache_root = resolve_path(str(paths_cfg.get("cache_root", "data/cache")), root=project_root())
        latent_path = latent_cache_path(stem, cache_root)
        save_global_latent(result.global_latent, latent_path)
        out["global_latent_path"] = latent_path

    return out
