from __future__ import annotations

import math
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
    lo: float = 30.0,
    hi: float = 50.0,
) -> float:
    """
    Clamp pitch (degrees above the XZ / ground plane in the Y-up convention) to ``[lo, hi]``.

    Callers should keep ``lo``/``hi`` inside the project policy band (default **30°–50°**): moderate
    downward views so interiors, rims, handles, and tops are visible — not horizon-grazing
    (``~0°``) or top-down.
    """
    if lo > hi:
        raise ValueError(f"elevation_min_deg ({lo}) must be <= elevation_max_deg ({hi})")
    return float(min(max(elevation_deg, lo), hi))


# Strict orbit policy: pitch (degrees above XZ) never leaves this band — avoids ~0° “flat”
# horizon views and extreme top-down; exposes tops, openings, and handles for VLMs / SAM.
POLICY_ELEVATION_MIN_DEG = 30.0
POLICY_ELEVATION_MAX_DEG = 50.0


def spherical_camera_poses(
    num_views: int,
    *,
    radius: float = 2.0,
    elevation_deg: float = 40.0,
    elevation_min_deg: float = 30.0,
    elevation_max_deg: float = 50.0,
    target: np.ndarray | None = None,
) -> list[CameraPose]:
    """
    Uniform azimuth (yaw) on a horizontal ring; pitch (elevation) is **fixed** for all views.

    The camera orbits at one elevation angle (``elevation_deg`` clamped after intersecting the
    requested band with ``[30°, 50°]`` above the XZ plane — enforced policy, not optional).
    Only azimuth changes between views.
    """
    if num_views < 1:
        raise ValueError("num_views must be >= 1")

    target = np.zeros(3, dtype=np.float64) if target is None else np.asarray(target, dtype=np.float64)
    raw_lo = float(elevation_min_deg)
    raw_hi = float(elevation_max_deg)
    el_lo = max(POLICY_ELEVATION_MIN_DEG, min(raw_lo, POLICY_ELEVATION_MAX_DEG))
    el_hi = min(POLICY_ELEVATION_MAX_DEG, max(raw_hi, POLICY_ELEVATION_MIN_DEG))
    if el_lo > el_hi:
        el_lo, el_hi = POLICY_ELEVATION_MIN_DEG, POLICY_ELEVATION_MAX_DEG
    elev_deg = clamp_elevation_deg(elevation_deg, lo=el_lo, hi=el_hi)
    elev = math.radians(elev_deg)
    y = radius * math.sin(elev)
    ring = radius * math.cos(elev)

    poses: list[CameraPose] = []
    for i in range(num_views):
        azimuth = 2.0 * math.pi * i / num_views
        eye = target + np.array([ring * math.cos(azimuth), y, ring * math.sin(azimuth)], dtype=np.float64)
        poses.append(look_at_pose(eye, target=target))
    return poses
