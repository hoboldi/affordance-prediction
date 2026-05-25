from __future__ import annotations

from typing import Any

from datasets.mesh_loading import MeshData
from rendering.mesh_renderer import MeshRenderConfig, MeshRenderer, RenderView


def build_render_config(cfg: dict[str, Any]) -> MeshRenderConfig:
    r = cfg.get("rendering", {})
    return MeshRenderConfig(
        image_size=int(r.get("image_size", 512)),
        fov_deg=float(r.get("fov_deg", 60.0)),
        num_views=int(r.get("num_views", 6)),
        camera_radius=float(r.get("camera_radius", 2.0)),
        elevation_deg=float(r.get("elevation_deg", 30.0)),
        depth_tolerance=float(r.get("depth_tolerance", 0.05)),
        depth_relative_tolerance=float(r.get("depth_relative_tolerance", 0.03)),
    )


def render_mesh_views(mesh: MeshData, cfg: dict[str, Any]) -> list[RenderView]:
    """Render novel views according to ``rendering.backend`` (mesh-only for now)."""
    backend = cfg.get("rendering", {}).get("backend", "mesh")
    if backend not in ("mesh", "both"):
        raise NotImplementedError(
            f"rendering.backend={backend!r} requires a full Gaussian **raster** pipeline. "
            "For multi-view **RGB from a 3DGS .ply** (point-centre preview), use "
            "`rendering.render_gaussian_splat_views` or `scripts/render_gaussian_views.py`. "
            "Use backend: mesh for mesh-only development."
        )
    renderer = MeshRenderer(build_render_config(cfg))
    return renderer.render(mesh)
