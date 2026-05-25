"""
True **3D Gaussian splat** rasterisation via the ``gsplat`` library (CUDA).

This is what produces object-like RGB from a standard Inria / Nerfstudio ``.ply`` (means, SH DC,
opacity, log-scales, quaternion). The pyrender path in ``gaussian_point_renderer`` only draws
Gaussian **centres** as points or icospheres — not ellipsoidal splatting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rendering.camera_sampling import spherical_camera_poses
from rendering.gaussian_ply import GaussianSplatPLY, load_gaussian_splat_ply, sh_dc_to_rgb, sigmoid
from rendering.mesh_renderer import MeshRenderConfig, RenderView


def _perspective_intrinsics(width: int, height: int, yfov: float) -> np.ndarray:
    fy = height / (2.0 * np.tan(yfov / 2.0))
    fx = fy * (width / height)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# Right-multiply camera axes: OpenGL (−Z forward) → OpenCV (+Z forward), same as common gsplat loaders.
_GL_TO_CV_ROT = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)


def _c2w_pyrender_gl_to_gsplat_opencv(c2w_gl: np.ndarray) -> np.ndarray:
    """
    ``spherical_camera_poses`` / pyrender use an OpenGL-style camera frame; ``gsplat`` rasterisation
    expects world-to-camera consistent with COLMAP / OpenCV (+Z into the scene). Without this flip,
    real 3DGS PLYs (e.g. AffordSplat) often render **fully black** while synthetic tests may still look fine.
    """
    m = np.asarray(c2w_gl, dtype=np.float64).copy()
    m[:3, :3] = m[:3, :3] @ _GL_TO_CV_ROT
    return m


def _subsample_ply(ply: GaussianSplatPLY, max_points: int, seed: int | None) -> GaussianSplatPLY:
    n = int(ply.means.shape[0])
    if n <= max_points:
        return ply
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return GaussianSplatPLY(
        means=ply.means[idx],
        f_dc=ply.f_dc[idx],
        opacity=ply.opacity[idx],
        scales=ply.scales[idx],
        rotations=ply.rotations[idx],
        rotation_layout=ply.rotation_layout,
        property_names=ply.property_names,
    )


def _normalize_means_and_log_scales(
    means: np.ndarray,
    scales_log: np.ndarray,
    *,
    target_radius: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``gaussian_point_renderer._normalize_points`` behaviour + isotropic log-scale shift."""
    c = means.mean(axis=0)
    centered = means.astype(np.float64) - c
    extent = float(np.linalg.norm(centered, axis=1).max())
    if extent < 1e-8:
        return means.astype(np.float32), scales_log.astype(np.float32)
    s = target_radius / extent
    means_out = (centered * s).astype(np.float32)
    log_s = float(np.log(s))
    scales_out = scales_log.astype(np.float32) + np.float32(log_s)
    return means_out, scales_out


def render_gaussian_splat_gsplat_views(
    ply_path: str | Path,
    config: MeshRenderConfig | None = None,
    *,
    max_points: int = 500_000,
    normalize_scene: bool = True,
    seed: int | None = None,
    device: torch.device | str | None = None,
    backgrounds: tuple[float, float, float] | None = None,
    radius_clip: float = 0.0,
    convert_pyrender_camera_to_gsplat: bool = True,
) -> list[RenderView]:
    """
    Rasterise a 3DGS ``.ply`` with ``gsplat`` for each camera in ``MeshRenderConfig``.

    **Requirements:** CUDA GPU and ``pip install gsplat`` (included in the SAM3D Docker ``[inference]`` stack).

    Cameras from ``spherical_camera_poses`` are **OpenGL / pyrender** style; ``gsplat`` expects a
    **COLMAP / OpenCV** camera frame for projection. By default we convert (``convert_pyrender_camera_to_gsplat=True``).
    Without this, many real 3DGS PLYs (e.g. AffordSplat) render **all black** while tiny test scenes may still look fine.
    Set ``convert_pyrender_camera_to_gsplat=False`` only if you know your PLY was authored for GL-style cameras.

    Colours use **degree-0 SH only** (view-independent), matching typical ``f_dc_*`` exports without
    ``f_rest`` evaluation per ray.
    """
    try:
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise ImportError(
            "Full Gaussian splat rendering needs the `gsplat` package. "
            "Example: `pip install gsplat` (SAM3D Docker: already in `.[inference]`). "
            "Until then, use `render_gaussian_splat_views` for a pyrender **centre preview** only."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "`gsplat` rasterization requires CUDA (`torch.cuda.is_available()` is False). "
            "Run on a GPU machine or use the point-centre preview renderer instead."
        )

    dev = torch.device(device) if device is not None else torch.device("cuda", 0)
    cfg = config or MeshRenderConfig()
    ply = load_gaussian_splat_ply(Path(ply_path))
    ply = _subsample_ply(ply, max_points, seed)

    means = ply.means.astype(np.float32)
    scales_log = ply.scales.astype(np.float32)
    if normalize_scene:
        means, scales_log = _normalize_means_and_log_scales(means, scales_log, target_radius=0.95)

    if ply.rotation_layout.lower() != "wxyz":
        raise ValueError(
            f"gsplat path expects quaternion layout 'wxyz' (Inria PLY); got {ply.rotation_layout!r}"
        )

    means_t = torch.from_numpy(means).to(dev)
    quats_t = torch.from_numpy(ply.rotations.astype(np.float32)).to(dev)
    scales_t = torch.exp(torch.from_numpy(scales_log).to(dev))
    opacities_t = torch.from_numpy(sigmoid(ply.opacity)).to(dev)
    rgb = sh_dc_to_rgb(ply.f_dc)
    colors_t = torch.from_numpy(rgb.astype(np.float32)).to(dev)

    poses = spherical_camera_poses(
        cfg.num_views,
        radius=cfg.camera_radius,
        elevation_deg=cfg.elevation_deg,
    )
    width = height = cfg.image_size
    yfov = float(np.deg2rad(cfg.fov_deg))
    K_np = _perspective_intrinsics(width, height, yfov)
    K_t = torch.from_numpy(K_np.astype(np.float32)).to(dev)

    c2w_np = [p.matrix.astype(np.float64) for p in poses]
    if convert_pyrender_camera_to_gsplat:
        c2w_np = [_c2w_pyrender_gl_to_gsplat_opencv(m) for m in c2w_np]
    c2w_list = [torch.from_numpy(m.astype(np.float32)).to(dev) for m in c2w_np]
    viewmats = torch.stack([torch.linalg.inv(m) for m in c2w_list], dim=0)
    Ks = K_t.unsqueeze(0).expand(viewmats.shape[0], -1, -1)

    backgrounds_arg: torch.Tensor | None = None
    if backgrounds is not None:
        bg = torch.tensor(backgrounds, dtype=torch.float32, device=dev).view(1, 1, 3)
        backgrounds_arg = bg.expand(1, viewmats.shape[0], 3).contiguous()

    with torch.no_grad():
        render_colors, _render_alphas, _meta = rasterization(
            means_t,
            quats_t,
            scales_t,
            opacities_t,
            colors_t,
            viewmats,
            Ks,
            width,
            height,
            render_mode="RGB+ED",
            backgrounds=backgrounds_arg,
            radius_clip=radius_clip,
            packed=False,
        )

    # gsplat returns [..., C, H, W, X]; drop a leading singleton batch if present.
    rc = render_colors
    if rc.dim() == 5:
        rc = rc[0]
    if rc.dim() != 4 or rc.shape[-1] < 4:
        raise RuntimeError(
            f"Unexpected gsplat render_colors shape after squeeze: {tuple(rc.shape)} "
            f"(raw {tuple(render_colors.shape)})"
        )

    views: list[RenderView] = []
    v_empty = np.zeros((0, 2), dtype=np.float32)
    vis_empty = np.zeros((0,), dtype=bool)
    c = int(rc.shape[0])
    for i in range(c):
        rgb_ch = rc[i, ..., :3].clamp(0.0, 1.0).detach().cpu().numpy()
        rgb_u8 = (rgb_ch * 255.0 + 0.5).astype(np.uint8)
        depth = rc[i, ..., 3].detach().cpu().numpy().astype(np.float32)
        views.append(
            RenderView(
                rgb=rgb_u8,
                depth=depth,
                vertex_uv=v_empty,
                vertex_visible=vis_empty,
                camera_pose=c2w_np[i],
                intrinsics=K_np,
            )
        )
    return views
