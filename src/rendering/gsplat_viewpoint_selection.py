"""
Select the best single 2D view from a 3DGS asset by aligning unprojected depth with a GT point cloud.

Pipeline (deterministic given ``seed``):

1. **Three fixed azimuths** on the ring at fixed elevation (default **−45°, 0°, +45°** vs the same
   zero direction as a uniform sweep’s first camera), instead of a dense multi-sample orbit. Set
   ``orbit_azimuth_offsets_deg=None`` and increase ``num_views`` for a **uniform** full-ring sweep.
   The ring’s pole is **world +Y**, then rotated by ``orbit_ring_rotation_deg`` about
   ``orbit_ring_rotation_axis`` (defaults **+90° about +X** for common splat vs world-up mismatch).
   Override with ``orbit_axis`` / ``orbit_axis_mode`` (``pca_min`` / ``pca_max``) if needed.
2. gsplat ``RGB+ED`` → depth; unproject with pinhole ``K`` (Z-buffer assumption) to camera space,
   then transform to the **normalized scene frame** used by gsplat (same center/scale as splat means).
3. Score each view with inverse-distance k-NN (higher = better) plus reject rules: by default only **low valid-depth fraction** and **non-finite Chamfer** (optional **strict** mode also rejects extreme covariance elongation and NN outlier rate — often misfires on unprojected depth vs sparse GT).
4. Return the best view RGB + metadata.

Optionally **upweight alignment to an affordance-region 3D point set** (e.g. AffordSplat
``<verb>/GS_anno_<id>.ply``) by adding ``affordance_score_weight * idw_knn(render, affordance)``
to the base GT score (same normalization frame as full GT).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rendering.camera_sampling import resolve_spherical_orbit_axis
from rendering.gaussian_gsplat_renderer import (
    _normalize_means_and_log_scales,
    _subsample_ply,
    render_gaussian_splat_gsplat_views,
)
from rendering.gaussian_ply import load_gaussian_splat_ply
from rendering.mesh_renderer import MeshRenderConfig, RenderView


def _scene_normalize_params(means: np.ndarray, *, target_radius: float = 0.95) -> tuple[np.ndarray, float]:
    """Return center ``c`` and isotropic scale ``s`` such that normalized means match ``_normalize_means_and_log_scales``."""
    c = means.astype(np.float64).mean(axis=0)
    centered = means.astype(np.float64) - c
    extent = float(np.linalg.norm(centered, axis=1).max())
    if extent < 1e-8:
        return c, 1.0
    s = target_radius / extent
    return c, s


def apply_scene_normalize(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (points.astype(np.float64) - center) * scale


def _view_azimuths_deg_list(cfg: ViewSelectionConfig) -> list[float]:
    """Ring azimuth (degrees) for each view index, matching ``spherical_camera_poses`` order."""
    if cfg.orbit_azimuth_offsets_deg is not None:
        return [float(x) for x in cfg.orbit_azimuth_offsets_deg]
    return [360.0 * i / cfg.num_views for i in range(cfg.num_views)]


def _pick_best_view_index(scores: list[float], chamfers: list[float]) -> int:
    """
    Argmax on combined score, with a deterministic tie-break on Chamfer (then lower index).

    Near-ties within a small relative band of the max score are treated as ties so ``argmax``
    does not arbitrarily prefer the lowest index when floats are almost equal.
    """
    scores_np = np.asarray(scores, dtype=np.float64)
    cham_np = np.asarray(chamfers, dtype=np.float64)
    finite_score = np.isfinite(scores_np) & (scores_np > float("-inf"))
    if not np.any(finite_score):
        return 0
    max_sc = float(np.max(scores_np[finite_score]))
    eps = max(1e-12, 1e-7 * max(1.0, abs(max_sc)))
    near = finite_score & (scores_np >= max_sc - eps)
    cand_idx = np.flatnonzero(near)
    if len(cand_idx) == 1:
        return int(cand_idx[0])
    sub_ch = cham_np[cand_idx].astype(np.float64)
    sub_ch[~np.isfinite(sub_ch)] = np.inf
    order = np.lexsort((cand_idx, sub_ch))
    return int(cand_idx[order[0]])


def unproject_depth_zbuffer(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    depth_valid_min: float = 1e-4,
    stride: int = 1,
) -> np.ndarray:
    """
    Unproject depth buffer to camera-space points (OpenCV: X right, Y down, Z forward).

    Assumes ``depth`` stores **metric Z** along the optical axis at each pixel (common for ED).
    """
    h, w = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us = np.arange(0, w, stride, dtype=np.float64)
    vs = np.arange(0, h, stride, dtype=np.float64)
    u, v = np.meshgrid(us, vs)
    z = depth[::stride, ::stride].astype(np.float64)
    valid = np.isfinite(z) & (z > depth_valid_min)
    u_v = u[valid]
    v_v = v[valid]
    z_v = z[valid]
    x = (u_v - cx) * z_v / fx
    y = (v_v - cy) * z_v / fy
    return np.stack([x, y, z_v], axis=1).astype(np.float32)


def transform_cam_to_world(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """``c2w`` is 4×4 camera-to-world (OpenCV-style, as stored on ``RenderView.camera_pose``)."""
    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    return (R @ points_cam.astype(np.float64).T).T + t


def load_gt_point_cloud(path: str | Path) -> np.ndarray:
    """Load ``(N, 3)`` float64 from ``.npy`` (``(N,3)``) or mesh vertex PLY/OBJ via trimesh."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"GT .npy must be (N, 3), got {arr.shape}")
        return arr.astype(np.float64)
    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.PointCloud):
        v = np.asarray(loaded.vertices, dtype=np.float64)
    else:
        mesh = loaded
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
        v = np.asarray(mesh.vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"GT point cloud has invalid vertices shape {v.shape}")
    if len(v) == 0:
        raise ValueError(f"GT file has no vertices (wrong loader or empty): {path}")
    return v


def _cov_elongation(points: np.ndarray) -> float:
    """Largest / smallest eigenvalue ratio of 3×3 covariance (>=1)."""
    if len(points) < 12:
        return 1.0
    c = points - points.mean(axis=0, keepdims=True)
    cov = (c.T @ c) / max(len(points) - 1, 1)
    w = np.linalg.eigvalsh(cov.astype(np.float64))
    w = np.clip(w, 1e-9, None)
    return float(w[-1] / w[0])


def idw_knn_score(
    p_render: np.ndarray,
    p_gt: np.ndarray,
    *,
    k: int = 3,
    eps: float = 2e-3,
    device: torch.device | None = None,
) -> float:
    """
    Mean over ``p_render`` of ``sum(1/(d_i+eps)) / k`` for the ``k`` nearest neighbours in ``p_gt``.

    Higher is better (tighter fit to GT surface).
    """
    if len(p_render) < 8 or len(p_gt) < 8:
        return float("-inf")
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.from_numpy(p_render).to(dev)
    b = torch.from_numpy(p_gt).to(dev)
    d = torch.cdist(a, b)
    knn_d, _ = torch.topk(d, k=min(k, d.shape[1]), largest=False, dim=1)
    score = (1.0 / (knn_d + eps)).mean().item()
    if not np.isfinite(score):
        return float("-inf")
    return float(score)


def one_way_chamfer_mean(
    p_render: np.ndarray,
    p_gt: np.ndarray,
    *,
    device: torch.device | None = None,
) -> float:
    """Mean squared min distance from each render point to GT (lower is better)."""
    if len(p_render) < 8 or len(p_gt) < 8:
        return float("inf")
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.from_numpy(p_render).to(dev)
    b = torch.from_numpy(p_gt).to(dev)
    d = torch.cdist(a, b)
    dmin = d.min(dim=1).values
    return float((dmin**2).mean().item())


@dataclass(frozen=True)
class ViewSelectionConfig:
    num_views: int = 3
    image_size: int = 512
    fov_deg: float = 60.0
    camera_radius: float = 2.0
    elevation_deg: float = 42.5
    elevation_min_deg: float = 25.0
    elevation_max_deg: float = 55.0
    max_gsplat_points: int = 500_000
    normalize_scene: bool = True
    unproject_stride: int = 2
    score_max_render: int = 4096
    score_max_gt: int = 8192
    min_valid_frac: float = 0.04
    max_elongation: float = 120.0
    max_outlier_frac: float = 0.68
    outlier_dist_thresh: float = 0.32
    # Unprojected depth points are often extremely elongated in world space; sparse GT (e.g. PC)
    # also breaks fixed NN outlier gates. Default False = rank all views with finite Chamfer only.
    use_strict_geometry_rejects: bool = False
    seed: int = 0
    # Camera orbit: default world +Y pole, then +90° about +X (common AffordSplat / gsplat frame fix).
    # Set ``orbit_ring_rotation_deg=0`` to disable; use ``orbit_axis`` / ``orbit_axis_mode`` for other fixes.
    orbit_axis: tuple[float, float, float] | None = None
    orbit_axis_mode: str = "world"
    orbit_ring_rotation_deg: float = 90.0
    orbit_ring_rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # Fixed azimuth samples (degrees) on the ring; ``None`` = uniform ``2π i / num_views`` sweep.
    orbit_azimuth_offsets_deg: tuple[float, ...] | None = (-45.0, 0.0, 45.0)
    # Optional affordance-region 3D points (same world frame as full GT before normalization).
    affordance_gt_path: str | Path | None = None
    affordance_score_weight: float = 0.0
    score_max_affordance: int = 4096

    def __post_init__(self) -> None:
        o = self.orbit_azimuth_offsets_deg
        if o is not None and len(o) != self.num_views:
            raise ValueError(
                "orbit_azimuth_offsets_deg length must match num_views "
                f"(got len={len(o)}, num_views={self.num_views}). "
                "Use orbit_azimuth_offsets_deg=None for a uniform full-ring sweep."
            )


@dataclass
class ViewSelectionResult:
    best_index: int
    scores: list[float]
    base_scores: list[float]
    affordance_scores: list[float]
    chamfers: list[float]
    rejected: list[str]
    all_views: list[RenderView]
    best_view: RenderView
    meta: dict[str, Any]


def select_best_gsplat_view_for_gt(
    splat_path: str | Path,
    gt_path: str | Path,
    cfg: ViewSelectionConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> ViewSelectionResult:
    """
    Render ``cfg.num_views`` gsplat orbit views, score against GT, return best RGB view.

    Cameras use ``cfg.orbit_azimuth_offsets_deg`` when set (fixed azimuths on the ring); otherwise
    a uniform azimuth sweep. **Requires** CUDA + ``gsplat`` (same as ``render_gaussian_splat_gsplat_views``).
    """
    cfg = cfg or ViewSelectionConfig()
    rng_gt = np.random.default_rng(cfg.seed + 901)
    rng_aff = np.random.default_rng(cfg.seed + 702)

    ply = load_gaussian_splat_ply(Path(splat_path))
    ply = _subsample_ply(ply, cfg.max_gsplat_points, cfg.seed)
    means = ply.means.astype(np.float32)
    center, scale = _scene_normalize_params(means, target_radius=0.95)
    gt_raw = load_gt_point_cloud(gt_path)
    gt_norm = apply_scene_normalize(gt_raw, center, scale)

    means_meta = ply.means.astype(np.float32)
    sc_meta = ply.scales.astype(np.float32)
    if cfg.normalize_scene:
        means_meta, _ = _normalize_means_and_log_scales(means_meta, sc_meta, target_radius=0.95)
    orbit_axis_used = resolve_spherical_orbit_axis(
        means_meta.astype(np.float64),
        orbit_axis=cfg.orbit_axis,
        orbit_axis_mode=cfg.orbit_axis_mode,
        ring_rotation_deg=cfg.orbit_ring_rotation_deg,
        ring_rotation_axis=cfg.orbit_ring_rotation_axis,
    )

    mrc = MeshRenderConfig(
        image_size=cfg.image_size,
        fov_deg=cfg.fov_deg,
        num_views=cfg.num_views,
        camera_radius=cfg.camera_radius,
        elevation_deg=cfg.elevation_deg,
        elevation_min_deg=cfg.elevation_min_deg,
        elevation_max_deg=cfg.elevation_max_deg,
        orbit_axis=cfg.orbit_axis,
        orbit_axis_mode=cfg.orbit_axis_mode,
        orbit_ring_rotation_deg=cfg.orbit_ring_rotation_deg,
        orbit_ring_rotation_axis=cfg.orbit_ring_rotation_axis,
        orbit_azimuth_offsets_deg=cfg.orbit_azimuth_offsets_deg,
        render_geometry_aux=False,
    )
    views = render_gaussian_splat_gsplat_views(
        splat_path,
        mrc,
        max_points=cfg.max_gsplat_points,
        normalize_scene=cfg.normalize_scene,
        seed=cfg.seed,
        device=device,
    )
    if len(views) != cfg.num_views:
        raise RuntimeError(f"Expected {cfg.num_views} views, got {len(views)}")

    # GT subsample for scoring (deterministic)
    if len(gt_norm) > cfg.score_max_gt:
        idx = rng_gt.choice(len(gt_norm), size=cfg.score_max_gt, replace=False)
        p_gt_score = gt_norm[idx].astype(np.float32)
    else:
        p_gt_score = gt_norm.astype(np.float32)

    p_aff_score: np.ndarray | None = None
    has_aff = False
    if cfg.affordance_score_weight > 0 and cfg.affordance_gt_path is not None:
        ap = Path(cfg.affordance_gt_path)
        if not ap.is_file():
            warnings.warn(f"affordance_gt_path not found ({ap}); affordance term disabled", stacklevel=2)
        else:
            aff_raw = load_gt_point_cloud(ap)
            aff_norm = apply_scene_normalize(aff_raw, center, scale)
            if len(aff_norm) < 8:
                warnings.warn(
                    f"Affordance point set has fewer than 8 points ({len(aff_norm)}); "
                    "affordance term disabled",
                    stacklevel=2,
                )
            else:
                if len(aff_norm) > cfg.score_max_affordance:
                    idx_a = rng_aff.choice(len(aff_norm), size=cfg.score_max_affordance, replace=False)
                    p_aff_score = aff_norm[idx_a].astype(np.float32)
                else:
                    p_aff_score = aff_norm.astype(np.float32)
                has_aff = True
    elif cfg.affordance_score_weight > 0 and cfg.affordance_gt_path is None:
        warnings.warn(
            "affordance_score_weight > 0 but affordance_gt_path is unset; using base GT score only",
            stacklevel=2,
        )

    scores: list[float] = []
    base_scores: list[float] = []
    affordance_scores: list[float] = []
    chamfers: list[float] = []
    rejected: list[str] = []

    dev = torch.device(device) if device is not None else None

    for i, view in enumerate(views):
        d = view.depth
        valid_frac = float(np.mean(np.isfinite(d) & (d > 1e-4)))
        reason_parts: list[str] = []
        if valid_frac < cfg.min_valid_frac:
            reason_parts.append(f"valid_frac={valid_frac:.3f}")

        pts_cam = unproject_depth_zbuffer(d, view.intrinsics, stride=cfg.unproject_stride)
        p_w = transform_cam_to_world(pts_cam, view.camera_pose).astype(np.float32)

        if len(p_w) > cfg.score_max_render:
            rs = np.random.default_rng(cfg.seed + 17_000 * i)
            idx2 = rs.choice(len(p_w), size=cfg.score_max_render, replace=False)
            p_rs = p_w[idx2]
        else:
            p_rs = p_w

        elong = _cov_elongation(p_rs)
        if cfg.use_strict_geometry_rejects and elong > cfg.max_elongation:
            reason_parts.append(f"elongation={elong:.1f}")

        cd = one_way_chamfer_mean(p_rs, p_gt_score, device=dev)
        if np.isfinite(cd):
            if cfg.use_strict_geometry_rejects:
                dists = np.linalg.norm(p_rs[:, None, :] - p_gt_score[None, :, :], axis=-1).min(axis=1)
                outlier_frac = float(np.mean(dists > cfg.outlier_dist_thresh))
                if outlier_frac > cfg.max_outlier_frac:
                    reason_parts.append(f"outlier_frac={outlier_frac:.2f}")
        else:
            reason_parts.append("chamfer_nan")

        if reason_parts:
            scores.append(float("-inf"))
            base_scores.append(float("-inf"))
            affordance_scores.append(float("nan"))
            chamfers.append(float("inf"))
            rejected.append(f"view_{i}: " + ", ".join(reason_parts))
            continue

        sc_base = idw_knn_score(p_rs, p_gt_score, k=3, device=dev)
        sc_aff = float("nan")
        if has_aff and p_aff_score is not None and cfg.affordance_score_weight > 0:
            sc_aff = idw_knn_score(p_rs, p_aff_score, k=3, device=dev)
            sc = sc_base + cfg.affordance_score_weight * sc_aff
        else:
            sc = sc_base
        scores.append(sc)
        base_scores.append(sc_base)
        affordance_scores.append(sc_aff)
        chamfers.append(cd)

    best_index = _pick_best_view_index(scores, chamfers)
    if not np.isfinite(scores[best_index]):
        raise RuntimeError(
            "All views rejected — typical fixes: lower min_valid_frac; ensure Chamfer is finite; "
            "if you enabled use_strict_geometry_rejects, try turning it off or relax "
            "max_outlier_frac / max_elongation. First reasons: "
            + "; ".join(rejected[:8])
        )

    best = views[best_index]
    meta = {
        "best_index": best_index,
        "score": scores[best_index],
        "score_base": base_scores[best_index],
        "score_affordance": affordance_scores[best_index],
        "affordance_score_weight": cfg.affordance_score_weight,
        "affordance_gt_path": str(Path(cfg.affordance_gt_path).resolve())
        if cfg.affordance_gt_path is not None
        else None,
        "chamfer_mean_sq": chamfers[best_index],
        "num_views": cfg.num_views,
        "elevation_deg": cfg.elevation_deg,
        "elevation_min_deg": cfg.elevation_min_deg,
        "elevation_max_deg": cfg.elevation_max_deg,
        "camera_radius": cfg.camera_radius,
        "image_size": cfg.image_size,
        "fov_deg": cfg.fov_deg,
        "seed": cfg.seed,
        "use_strict_geometry_rejects": cfg.use_strict_geometry_rejects,
        "orbit_axis_mode": cfg.orbit_axis_mode,
        "orbit_axis": list(cfg.orbit_axis) if cfg.orbit_axis is not None else None,
        "orbit_axis_used": orbit_axis_used.tolist(),
        "orbit_ring_rotation_deg": cfg.orbit_ring_rotation_deg,
        "orbit_ring_rotation_axis": list(cfg.orbit_ring_rotation_axis),
        "orbit_azimuth_offsets_deg": list(cfg.orbit_azimuth_offsets_deg)
        if cfg.orbit_azimuth_offsets_deg is not None
        else None,
        "view_azimuth_deg": _view_azimuths_deg_list(cfg),
        # Same isotropic normalize as scoring / gsplat render (center + scale from splat means).
        "scene_normalize_center": center.astype(float).tolist(),
        "scene_normalize_scale": float(scale),
        "splat_path": str(Path(splat_path).resolve()),
        "gt_path": str(Path(gt_path).resolve()),
        "scores": scores,
        "base_scores": base_scores,
        "affordance_scores": affordance_scores,
        "chamfers": chamfers,
        "rejected": rejected,
    }
    return ViewSelectionResult(
        best_index=best_index,
        scores=scores,
        base_scores=base_scores,
        affordance_scores=affordance_scores,
        chamfers=chamfers,
        rejected=rejected,
        all_views=views,
        best_view=best,
        meta=meta,
    )


def _meta_json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats so ``json.dumps(..., allow_nan=False)`` succeeds."""
    if isinstance(obj, dict):
        return {str(k): _meta_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_meta_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def export_best_view_png(
    result: ViewSelectionResult,
    output_png: str | Path,
    meta_json: str | Path | None = None,
) -> None:
    """Write best RGB as PNG and optional JSON sidecar with ``result.meta`` + camera matrices."""
    from PIL import Image

    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.best_view.rgb).save(out)

    if meta_json is not None:
        mp = Path(meta_json)
        mp.parent.mkdir(parents=True, exist_ok=True)
        serializable = _meta_json_safe(dict(result.meta))
        serializable["camera_to_world"] = result.best_view.camera_pose.astype(float).tolist()
        serializable["intrinsics"] = result.best_view.intrinsics.astype(float).tolist()
        mp.write_text(json.dumps(serializable, indent=2, allow_nan=False))
