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


def spherical_camera_poses(
    num_views: int,
    *,
    radius: float = 2.0,
    elevation_deg: float = 30.0,
    target: np.ndarray | None = None,
) -> list[CameraPose]:
    """
    Uniform azimuth on a cone at fixed elevation (degrees above the XZ plane).

    Assumes the mesh is centered at the origin.
    """
    if num_views < 1:
        raise ValueError("num_views must be >= 1")

    target = np.zeros(3, dtype=np.float64) if target is None else np.asarray(target, dtype=np.float64)
    elev = math.radians(elevation_deg)
    y = radius * math.sin(elev)
    ring = radius * math.cos(elev)

    poses: list[CameraPose] = []
    for i in range(num_views):
        azimuth = 2.0 * math.pi * i / num_views
        eye = target + np.array([ring * math.cos(azimuth), y, ring * math.sin(azimuth)], dtype=np.float64)
        poses.append(look_at_pose(eye, target=target))
    return poses
