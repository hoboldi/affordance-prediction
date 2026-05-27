from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

from datasets.mesh_loading import MeshData
from rendering.mesh_renderer import MeshRenderConfig, MeshRenderer, RenderView

logger = logging.getLogger(__name__)


def build_render_config(cfg: dict[str, Any]) -> MeshRenderConfig:
    r = cfg.get("rendering", {})
    oa = r.get("orbit_axis")
    orbit_axis: tuple[float, float, float] | None = None
    if isinstance(oa, (list, tuple)) and len(oa) == 3:
        orbit_axis = (float(oa[0]), float(oa[1]), float(oa[2]))
    rot_ax_raw = r.get("orbit_ring_rotation_axis", (1.0, 0.0, 0.0))
    if isinstance(rot_ax_raw, (list, tuple)) and len(rot_ax_raw) == 3:
        rot_axis = (float(rot_ax_raw[0]), float(rot_ax_raw[1]), float(rot_ax_raw[2]))
    else:
        rot_axis = (1.0, 0.0, 0.0)
    return MeshRenderConfig(
        image_size=int(r.get("image_size", 512)),
        fov_deg=float(r.get("fov_deg", 60.0)),
        num_views=int(r.get("num_views", 6)),
        camera_radius=float(r.get("camera_radius", 2.0)),
        elevation_deg=float(r.get("elevation_deg", 42.5)),
        elevation_min_deg=float(r.get("elevation_min_deg", 25.0)),
        elevation_max_deg=float(r.get("elevation_max_deg", 60.0)),
        orbit_axis=orbit_axis,
        orbit_axis_mode=str(r.get("orbit_axis_mode", "world")),
        orbit_ring_rotation_deg=float(r.get("orbit_ring_rotation_deg", 0.0)),
        orbit_ring_rotation_axis=rot_axis,
        depth_tolerance=float(r.get("depth_tolerance", 0.05)),
        depth_relative_tolerance=float(r.get("depth_relative_tolerance", 0.03)),
        render_geometry_aux=bool(r.get("render_geometry_aux", True)),
    )


def _splat_rgb_views(ply_path: Path, mrc: MeshRenderConfig, r: dict[str, Any]) -> list[RenderView]:
    """Rasterise splat RGB with gsplat when possible, else pyrender centre preview."""
    max_points = int(r.get("splat_max_points", 500_000))
    seed = r.get("splat_seed")
    if seed is not None:
        seed = int(seed)
    normalize = bool(r.get("splat_normalize_scene", True))
    try:
        from rendering.gaussian_gsplat_renderer import render_gaussian_splat_gsplat_views

        return render_gaussian_splat_gsplat_views(
            ply_path,
            mrc,
            max_points=max_points,
            normalize_scene=normalize,
            seed=seed,
        )
    except (ImportError, RuntimeError) as exc:
        logger.info("Gaussian splat raster fallback (%s); using pyrender preview.", exc)
        from rendering.gaussian_point_renderer import render_gaussian_splat_views

        return render_gaussian_splat_views(
            ply_path,
            mrc,
            max_points=min(max_points, 200_000),
            normalize_scene=normalize,
            seed=seed if seed is not None else 0,
        )


def render_mesh_views(mesh: MeshData, cfg: dict[str, Any]) -> list[RenderView]:
    """
    Render novel views for the training / VLM pipeline.

    ``rendering.backend``:

    * ``mesh`` — pyrender mesh RGB + depth + per-vertex screen correspondences.
    * ``gaussian`` / ``both`` — same mesh pass for depth and ``vertex_uv`` / ``vertex_visible``,
      then replace **RGB** from ``rendering.splat_path`` when that file exists (gsplat if CUDA +
      ``gsplat`` is available, otherwise point-centre preview). If ``splat_path`` is missing or
      invalid, falls back to mesh RGB with a warning.
    """
    r = cfg.get("rendering", {})
    backend = r.get("backend", "mesh")
    mrc = build_render_config(cfg)
    mesh_views = MeshRenderer(mrc).render(mesh)

    if backend == "mesh":
        return mesh_views

    if backend not in ("gaussian", "both"):
        raise ValueError(
            f"rendering.backend must be mesh, gaussian, or both; got {backend!r}."
        )

    splat_raw = r.get("splat_path")
    if not splat_raw:
        warnings.warn(
            f"rendering.backend={backend!r} but splat_path is unset; using mesh RGB only.",
            UserWarning,
            stacklevel=2,
        )
        return mesh_views

    splat_path = Path(str(splat_raw)).expanduser()
    if not splat_path.is_file():
        warnings.warn(
            f"rendering.backend={backend!r} but splat_path is not a file ({splat_path}); "
            "using mesh RGB only.",
            UserWarning,
            stacklevel=2,
        )
        return mesh_views

    try:
        splat_views = _splat_rgb_views(splat_path, mrc, r)
    except Exception as exc:  # noqa: BLE001 — surface any PLY / pyrender failure as fallback
        warnings.warn(
            f"Splat RGB render failed ({exc!r}); using mesh RGB only.",
            UserWarning,
            stacklevel=2,
        )
        return mesh_views

    if len(splat_views) != len(mesh_views):
        raise RuntimeError(
            f"Splat produced {len(splat_views)} views but mesh produced {len(mesh_views)}; "
            "check rendering config consistency."
        )

    return [
        replace(mv, rgb=sv.rgb) for mv, sv in zip(mesh_views, splat_views, strict=True)
    ]
