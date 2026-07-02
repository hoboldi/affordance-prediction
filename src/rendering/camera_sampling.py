from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """Camera-to-world transform (4×4) with a look-at pose on the Y-up convention."""

    matrix: np.ndarray  # (4, 4) float64
    eye: np.ndarray  # (3,)
    target: np.ndarray  # (3,)
    up: np.ndarray  # (3,)


def look_at_pose(
    eye: np.ndarray,
    target: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> CameraPose:
    """Build a camera-to-world matrix (OpenGL / pyrender: camera looks down −Z)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.zeros(3, dtype=np.float64) if target is None else np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("eye and target are too close")
    forward /= norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right /= right_norm

    cam_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = cam_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return CameraPose(matrix=pose, eye=eye.copy(), target=target.copy(), up=cam_up.copy())


def clamp_elevation_deg(
    elevation_deg: float,
    *,
    lo: float = 25.0,
    hi: float = 60.0,
) -> float:
    """
    Clamp polar angle (degrees from the azimuth **ring plane** toward ``+orbit_axis``) to ``[lo, hi]``.

    Callers should keep ``lo``/``hi`` inside the project policy band (default **25°–60°**): broader
    downward-to-oblique coverage than a narrow band, while still avoiding horizon-grazing (~0°)
    and near–top-down extremes.
    """
    if lo > hi:
        raise ValueError(f"elevation_min_deg ({lo}) must be <= elevation_max_deg ({hi})")
    return float(min(max(elevation_deg, lo), hi))


# Strict orbit policy: polar angle (from the azimuth ring plane toward +orbit_axis) never leaves
# this band — avoids ~0° “flat” horizon views and extreme top-down; exposes tops, openings, and handles for VLMs / SAM.
# Defaults preserve the production band (25–60°). Override via env for wider-coverage experiments
# (e.g. Tier-2 lower-hemisphere re-extraction): AFFORD_ELEV_MIN_DEG / AFFORD_ELEV_MAX_DEG.
POLICY_ELEVATION_MIN_DEG = float(os.environ.get("AFFORD_ELEV_MIN_DEG", "25.0"))
POLICY_ELEVATION_MAX_DEG = float(os.environ.get("AFFORD_ELEV_MAX_DEG", "60.0"))


def orbit_plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthonormal ``u, v`` spanning the plane perpendicular to unit ``axis``.

    Azimuth ``θ`` uses ``cos(θ) u + sin(θ) v`` so cameras sweep a ring around ``axis``.
    """
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise ValueError("orbit axis must be non-zero")
    a = a / n
    if abs(a[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(ref, a)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u = np.cross(ref, a)
        u_norm = float(np.linalg.norm(u))
    u = u / u_norm
    v = np.cross(a, u)
    return u, v


def infer_orbit_axis_from_points(points: np.ndarray, *, mode: str) -> np.ndarray:
    """
    Unit vector **perpendicular to the azimuth ring** (same role as +Y in the classic XZ ring).

    ``mode``:
        - ``world`` — ``[0, 1, 0]`` (scene gravity up).
        - ``pca_min`` — PCA axis with **smallest** spread (often the thin / “stacking” direction).
        - ``pca_max`` — PCA axis with **largest** spread (elongation direction).
    """
    m = (mode or "world").lower().strip()
    if m == "world":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x = pts - pts.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if m == "pca_min":
        a = vh[2].astype(np.float64)
    elif m == "pca_max":
        a = vh[0].astype(np.float64)
    else:
        raise ValueError(f"infer_orbit_axis_from_points: unknown mode {mode!r}")
    na = float(np.linalg.norm(a))
    if na < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    a = a / na
    if float(np.dot(a, np.array([0.0, 1.0, 0.0]))) < 0.0:
        a = -a
    return a


def rotate_vector_about_axis(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotate vector ``v`` (3,) around unit direction ``axis`` by ``deg`` degrees (right-hand rule)."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    k = np.asarray(axis, dtype=np.float64).reshape(3)
    nk = float(np.linalg.norm(k))
    if nk < 1e-12:
        return v
    k = k / nk
    if abs(deg) < 1e-12:
        return v
    theta = math.radians(float(deg))
    ct, st = math.cos(theta), math.sin(theta)
    return v * ct + np.cross(k, v) * st + k * float(np.dot(k, v)) * (1.0 - ct)


def rotation_matrix_from_axis_angle_deg(
    axis: tuple[float, ...] | np.ndarray,
    degrees: float,
) -> np.ndarray:
    """
    Return a proper rotation matrix ``R`` (3×3, float64) with ``det(R) = +1``.

    Column vectors transform as ``p_rotated = R @ p`` (same right-hand rule as
    :func:`rotate_vector_about_axis`).
    """
    k = np.asarray(axis, dtype=np.float64).reshape(3)
    nk = float(np.linalg.norm(k))
    if nk < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = k / nk
    if abs(degrees) < 1e-12:
        return np.eye(3, dtype=np.float64)
    theta = math.radians(float(degrees))
    ct, st = math.cos(theta), math.sin(theta)
    vx = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + st * vx + (1.0 - ct) * (vx @ vx)


def resolve_spherical_orbit_axis(
    reference_points: np.ndarray | None,
    *,
    orbit_axis: tuple[float, float, float] | None,
    orbit_axis_mode: str,
    ring_rotation_deg: float = 0.0,
    ring_rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Explicit ``orbit_axis`` wins; else infer from ``reference_points`` using ``orbit_axis_mode``.

    ``ring_rotation_deg`` / ``ring_rotation_axis`` apply a **fixed** Rodrigues rotation to the
    resolved pole (e.g. +90° about X to correct a systematically tilted asset frame vs world +Y).
    """
    if orbit_axis is not None:
        a = np.asarray(orbit_axis, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(a))
        if n < 1e-9:
            raise ValueError("orbit_axis must be a non-zero 3-vector")
        a = a / n
    else:
        mode = (orbit_axis_mode or "world").lower().strip()
        if mode == "world":
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif reference_points is None:
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            a = infer_orbit_axis_from_points(reference_points, mode=mode)

    if abs(ring_rotation_deg) > 1e-12:
        a = rotate_vector_about_axis(a, np.asarray(ring_rotation_axis, dtype=np.float64), ring_rotation_deg)
    n2 = float(np.linalg.norm(a))
    if n2 < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return a / n2


def spherical_camera_poses(
    num_views: int,
    *,
    radius: float = 2.0,
    elevation_deg: float = 42.5,
    elevation_min_deg: float = 25.0,
    elevation_max_deg: float = 60.0,
    target: np.ndarray | None = None,
    orbit_axis: np.ndarray | None = None,
    azimuth_offsets_deg: tuple[float, ...] | None = None,
) -> list[CameraPose]:
    """
    Cameras on one elevation ring around ``target``.

    * If ``azimuth_offsets_deg`` is ``None`` (default): **uniform** azimuth,
      ``2π·i/num_views`` for ``i = 0 … num_views-1`` (full ring sweep).
    * If set: one camera per entry, at that azimuth (degrees) in the ring plane — same zero
      direction as the first uniform sample. ``len(azimuth_offsets_deg)`` must equal ``num_views``.

    ``orbit_axis`` (unit, default ``[0,1,0]``) is the direction of **polar offset** from the ring
    plane (the classic case: ring in XZ, cameras raised along +Y). Resolve it with
    :func:`resolve_spherical_orbit_axis` (optional PCA modes + fixed ring rotation).
    """
    if num_views < 1:
        raise ValueError("num_views must be >= 1")
    if azimuth_offsets_deg is not None:
        if len(azimuth_offsets_deg) != num_views:
            raise ValueError(
                f"azimuth_offsets_deg length ({len(azimuth_offsets_deg)}) must equal num_views ({num_views})"
            )

    target = np.zeros(3, dtype=np.float64) if target is None else np.asarray(target, dtype=np.float64)
    raw_lo = float(elevation_min_deg)
    raw_hi = float(elevation_max_deg)
    el_lo = max(POLICY_ELEVATION_MIN_DEG, min(raw_lo, POLICY_ELEVATION_MAX_DEG))
    el_hi = min(POLICY_ELEVATION_MAX_DEG, max(raw_hi, POLICY_ELEVATION_MIN_DEG))
    if el_lo > el_hi:
        el_lo, el_hi = POLICY_ELEVATION_MIN_DEG, POLICY_ELEVATION_MAX_DEG
    elev_deg = clamp_elevation_deg(elevation_deg, lo=el_lo, hi=el_hi)
    elev = math.radians(elev_deg)
    pole = radius * math.sin(elev)
    ring_r = radius * math.cos(elev)

    if orbit_axis is None:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        a = np.asarray(orbit_axis, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(a))
        if n < 1e-9:
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            a = a / n
    u, v = orbit_plane_basis(a)

    if azimuth_offsets_deg is None:
        azimuths = [2.0 * math.pi * i / num_views for i in range(num_views)]
    else:
        azimuths = [math.radians(float(d)) for d in azimuth_offsets_deg]

    poses: list[CameraPose] = []
    for azimuth in azimuths:
        offset = ring_r * (math.cos(azimuth) * u + math.sin(azimuth) * v) + pole * a
        eye = target + offset.astype(np.float64)
        poses.append(look_at_pose(eye, target=target, up=a))
    return poses


def multi_ring_camera_poses(
    azimuths_per_ring: int,
    elevations_deg: tuple[float, ...],
    *,
    radius: float = 2.0,
    target: np.ndarray | None = None,
    orbit_axis: np.ndarray | None = None,
    elevation_min_deg: float = 25.0,
    elevation_max_deg: float = 60.0,
) -> list[CameraPose]:
    """Cameras on **multiple** elevation rings, with azimuths **staggered** across rings.

    The single-ring sampler bakes an N-fold azimuthal seam into projected per-vertex features (all
    cameras share one elevation, evenly spaced in azimuth). Spreading cameras over several elevations
    and interleaving their azimuths (ring k offset by ``k·360/(azimuths_per_ring·K)``) gives a set of
    well-distributed viewpoints that wash that seam out under mean/weighted fusion and cover more of
    the surface. Total views = ``azimuths_per_ring · len(elevations_deg)``.
    """
    if azimuths_per_ring < 1:
        raise ValueError("azimuths_per_ring must be >= 1")
    if not elevations_deg:
        raise ValueError("elevations_deg must be non-empty")
    K = len(elevations_deg)
    stagger = 360.0 / (azimuths_per_ring * K)
    base = [360.0 * i / azimuths_per_ring for i in range(azimuths_per_ring)]
    poses: list[CameraPose] = []
    for k, elev in enumerate(elevations_deg):
        offsets = tuple(b + k * stagger for b in base)
        poses.extend(
            spherical_camera_poses(
                azimuths_per_ring,
                radius=radius,
                elevation_deg=float(elev),
                elevation_min_deg=elevation_min_deg,
                elevation_max_deg=elevation_max_deg,
                target=target,
                orbit_axis=orbit_axis,
                azimuth_offsets_deg=offsets,
            )
        )
    return poses
