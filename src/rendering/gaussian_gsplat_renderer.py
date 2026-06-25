"""
True **3D Gaussian splat** rasterisation via the ``gsplat`` library (CUDA).

This is what produces object-like RGB from a standard Inria / Nerfstudio ``.ply`` (means, SH DC,
opacity, log-scales, quaternion). The pyrender path in ``gaussian_point_renderer`` only draws
Gaussian **centres** as points or icospheres — not ellipsoidal splatting.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from rendering.camera_sampling import resolve_spherical_orbit_axis, spherical_camera_poses
from rendering.gaussian_ply import GaussianSplatPLY, load_gaussian_splat_ply, sigmoid
from rendering.mesh_renderer import MeshRenderConfig, RenderView


def _sh_degree_cap_from_env() -> int | None:
    """Optional ``AFFORDANCE_GSPLAT_SH_DEGREE_CAP`` (0–3) to limit view-dependent SH in gsplat."""
    raw = os.environ.get("AFFORDANCE_GSPLAT_SH_DEGREE_CAP", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        return None
    return max(0, min(v, 4))


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


def _subsample_mode_from_env() -> Literal["uniform", "stratified"]:
    raw = os.environ.get("AFFORDANCE_GSPLAT_SUBSAMPLE", "stratified").strip().lower()
    if raw in ("uniform", "stratified"):
        return raw  # type: ignore[return-value]
    return "stratified"


def _ply_index_subset(ply: GaussianSplatPLY, idx: np.ndarray) -> GaussianSplatPLY:
    return GaussianSplatPLY(
        means=ply.means[idx],
        f_dc=ply.f_dc[idx],
        opacity=ply.opacity[idx],
        scales=ply.scales[idx],
        rotations=ply.rotations[idx],
        rotation_layout=ply.rotation_layout,
        property_names=ply.property_names,
        sh_coeffs=ply.sh_coeffs[idx] if ply.sh_coeffs is not None else None,
        sh_degree=ply.sh_degree,
    )


def _subsample_ply_uniform(ply: GaussianSplatPLY, max_points: int, seed: int | None) -> GaussianSplatPLY:
    n = int(ply.means.shape[0])
    if n <= max_points:
        return ply
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return _ply_index_subset(ply, idx)


def _subsample_ply_stratified(ply: GaussianSplatPLY, max_points: int, seed: int | None) -> GaussianSplatPLY:
    """
    Voxel-stratified subsample in **original** mean space so every occupied region of the bbox
    contributes Gaussians. Uniform random subsampling often keeps rare high-contrast splats (e.g.
    metal knobs) while missing the bulk of the object → grey mass with coloured flecks.
    """
    n = int(ply.means.shape[0])
    if n <= max_points:
        return ply
    rng = np.random.default_rng(seed)
    means = ply.means.astype(np.float64)
    lo = means.min(axis=0)
    hi = means.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    # ~4 Gaussians per occupied cell on average; grid capped for speed / memory.
    g = int(np.clip(np.ceil(np.cbrt(max(max_points // 4, 1))), 8, 48))
    coords = np.floor((means - lo) / span * (g - 1e-9)).clip(0.0, g - 1.0).astype(np.int32)
    cell = (coords[:, 0].astype(np.int64) * g + coords[:, 1]) * g + coords[:, 2]
    order = np.argsort(cell, kind="mergesort")
    cs = cell[order]
    if n == 1:
        seg0 = np.array([0], dtype=np.int64)
        seg1 = np.array([1], dtype=np.int64)
    else:
        diff = cs[1:] != cs[:-1]
        seg0 = np.concatenate([[0], np.flatnonzero(diff) + 1])
        seg1 = np.concatenate([seg0[1:], np.array([n], dtype=np.int64)])
    m = int(seg0.shape[0])
    chosen: list[int] = []
    if m >= max_points:
        perm = rng.permutation(m)
        for t in range(max_points):
            bi = int(perm[t])
            s, e = int(seg0[bi]), int(seg1[bi])
            pool = order[s:e]
            chosen.append(int(rng.choice(pool)))
    else:
        quota = max_points // m
        rem = max_points % m
        perm = rng.permutation(m)
        for t in range(m):
            bi = int(perm[t])
            s, e = int(seg0[bi]), int(seg1[bi])
            pool = order[s:e]
            k = min(e - s, quota + (1 if t < rem else 0))
            if k <= 0:
                continue
            take = rng.choice(pool, size=k, replace=False) if k < (e - s) else pool
            chosen.extend(int(x) for x in take.ravel().tolist())
    idx_u = np.unique(np.asarray(chosen, dtype=np.int64))
    if int(idx_u.shape[0]) < max_points:
        remain = np.setdiff1d(np.arange(n, dtype=np.int64), idx_u, assume_unique=False)
        need = max_points - int(idx_u.shape[0])
        if remain.size > 0:
            extra = rng.choice(remain, size=min(need, int(remain.size)), replace=False)
            idx_u = np.unique(np.concatenate([idx_u, extra]))
    if int(idx_u.shape[0]) > max_points:
        idx_u = rng.choice(idx_u, size=max_points, replace=False)
    return _ply_index_subset(ply, idx_u)


def _subsample_ply(
    ply: GaussianSplatPLY,
    max_points: int,
    seed: int | None,
    *,
    mode: Literal["uniform", "stratified"] | None = None,
) -> GaussianSplatPLY:
    m = mode or _subsample_mode_from_env()
    if m == "uniform":
        return _subsample_ply_uniform(ply, max_points, seed)
    return _subsample_ply_stratified(ply, max_points, seed)


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
    subsample_mode: Literal["uniform", "stratified"] | None = None,
) -> list[RenderView]:
    """
    Rasterise a 3DGS ``.ply`` with ``gsplat`` for each camera in ``MeshRenderConfig``.

    **Requirements:** CUDA GPU and ``pip install gsplat`` (included in the SAM3D Docker ``[inference]`` stack).

    Cameras from ``spherical_camera_poses`` are **OpenGL / pyrender** style; ``gsplat`` expects a
    **COLMAP / OpenCV** camera frame for projection. By default we convert (``convert_pyrender_camera_to_gsplat=True``).
    Without this, many real 3DGS PLYs (e.g. AffordSplat) render **all black** while tiny test scenes may still look fine.
    Set ``convert_pyrender_camera_to_gsplat=False`` only if you know your PLY was authored for GL-style cameras.

    Colours: when ``f_rest_*`` are present in the PLY, **full spherical harmonics** are passed to
    ``gsplat`` (``colors`` shaped ``(N, K, 3)`` with ``sh_degree``) so appearance is **view-dependent**
    like standard Inria / Nerfstudio training. If only ``f_dc_*`` exist, **degree-0 SH** is used
    (still ``gsplat``'s SH path, equivalent to view-independent diffuse from DC).
    Non-DC column order follows Inria **channel-major** flattening by default; set environment
    variable ``AFFORDANCE_GSPLAT_F_REST_LAYOUT=rgb_interleaved`` if a third-party PLY uses RGB
    triplets per SH band instead (wrong layout often looks **washed out / gray** in ``gsplat``).

    **Sparse / speckly colour:** By default we **voxel-stratify** subsampling (``AFFORDANCE_GSPLAT_SUBSAMPLE=stratified``)
    so Gaussians are taken from all occupied regions of the bbox — uniform random subsampling often
    oversamples rare high-contrast geometry (e.g. metal knobs) and undersamples large surfaces,
    which looks like a **grey object with coloured accents**. Set ``AFFORDANCE_GSPLAT_SUBSAMPLE=uniform``
    for the old behaviour, or raise ``max_points`` if you still need more detail.
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
    ply_src = load_gaussian_splat_ply(Path(ply_path))
    n_src = int(ply_src.means.shape[0])
    if n_src > max_points:
        sm = subsample_mode or _subsample_mode_from_env()
        warnings.warn(
            f"gsplat: subsampling {n_src} → {max_points} Gaussians (mode={sm!r}). "
            "If previews look grey with only small coloured details, increase max_points; "
            "with mode='uniform' rare high-contrast splats are oversampled.",
            UserWarning,
            stacklevel=2,
        )
    ply = _subsample_ply(ply_src, max_points, seed, mode=subsample_mode)

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
    if ply.sh_coeffs is not None and ply.sh_degree is not None:
        sh_coeffs = ply.sh_coeffs.astype(np.float32, copy=False)
        if sh_coeffs.shape[0] != means.shape[0]:
            raise ValueError("sh_coeffs row count must match means")
        if sh_coeffs.shape[1] != (ply.sh_degree + 1) ** 2:
            raise ValueError("sh_coeffs band count does not match sh_degree")
        colors_t = torch.from_numpy(sh_coeffs).to(dev)
        cap = _sh_degree_cap_from_env()
        eff_deg = int(ply.sh_degree) if cap is None else min(int(ply.sh_degree), cap)
        sh_deg_arg: int | None = eff_deg
    else:
        dc_only = ply.f_dc[:, np.newaxis, :].astype(np.float32, copy=False)
        colors_t = torch.from_numpy(dc_only).to(dev)
        sh_deg_arg = 0

    axis = resolve_spherical_orbit_axis(
        means.astype(np.float64),
        orbit_axis=cfg.orbit_axis,
        orbit_axis_mode=cfg.orbit_axis_mode,
        ring_rotation_deg=cfg.orbit_ring_rotation_deg,
        ring_rotation_axis=cfg.orbit_ring_rotation_axis,
    )
    poses = spherical_camera_poses(
        cfg.num_views,
        radius=cfg.camera_radius,
        elevation_deg=cfg.elevation_deg,
        elevation_min_deg=cfg.elevation_min_deg,
        elevation_max_deg=cfg.elevation_max_deg,
        orbit_axis=axis,
        azimuth_offsets_deg=cfg.orbit_azimuth_offsets_deg,
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
        bg = torch.tensor(backgrounds, dtype=torch.float32, device=dev).view(1, 3)
        backgrounds_arg = bg.expand(viewmats.shape[0], 3).contiguous()

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
            sh_degree=sh_deg_arg,
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
