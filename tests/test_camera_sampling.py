from __future__ import annotations

import math

import numpy as np
import pytest

from rendering.camera_sampling import (
    clamp_elevation_deg,
    infer_orbit_axis_from_points,
    rotate_vector_about_axis,
    rotation_matrix_from_axis_angle_deg,
    spherical_camera_poses,
)


def test_rotation_matrix_matches_rotate_vector_about_axis() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    for deg in (-37.0, 0.0, 45.0, 90.0):
        vr = rotate_vector_about_axis(v, axis, deg)
        R = rotation_matrix_from_axis_angle_deg(axis, deg)
        vr2 = R @ v
        assert np.linalg.norm(vr - vr2) < 1e-9
    R90 = rotation_matrix_from_axis_angle_deg(axis, 90.0)
    Rm90 = rotation_matrix_from_axis_angle_deg(axis, -90.0)
    assert np.linalg.norm(Rm90 @ R90 - np.eye(3)) < 1e-9


def test_clamp_elevation_deg() -> None:
    assert clamp_elevation_deg(5.0) == 25.0
    assert clamp_elevation_deg(90.0) == 60.0
    assert clamp_elevation_deg(40.0) == 40.0


def test_spherical_poses_fixed_elevation_varying_azimuth() -> None:
    poses = spherical_camera_poses(
        6,
        radius=2.0,
        elevation_deg=42.5,
        elevation_min_deg=25.0,
        elevation_max_deg=60.0,
    )
    ys = [float(p.eye[1]) for p in poses]
    assert all(abs(y - ys[0]) < 1e-9 for y in ys), "all views share the same height (fixed pitch)"
    rings = [math.hypot(float(p.eye[0]), float(p.eye[2])) for p in poses]
    assert all(abs(r - rings[0]) < 1e-9 for r in rings), "same horizontal ring radius"


def test_spherical_poses_azimuth_offsets_deg() -> None:
    poses = spherical_camera_poses(
        2,
        radius=1.0,
        elevation_deg=40.0,
        elevation_min_deg=25.0,
        elevation_max_deg=60.0,
        azimuth_offsets_deg=(0.0, 90.0),
    )
    assert len(poses) == 2

    def xz_dir(p) -> tuple[float, float]:
        x, z = float(p.eye[0]), float(p.eye[2])
        h = math.hypot(x, z)
        return x / h, z / h

    u0 = xz_dir(poses[0])
    u1 = xz_dir(poses[1])
    dot = u0[0] * u1[0] + u0[1] * u1[1]
    assert abs(dot) < 1e-5, "90° azimuth separation in the ring plane"


def test_spherical_poses_offsets_len_must_match_num_views() -> None:
    with pytest.raises(ValueError, match="azimuth_offsets_deg length"):
        spherical_camera_poses(
            3,
            radius=1.0,
            elevation_deg=40.0,
            elevation_min_deg=25.0,
            elevation_max_deg=60.0,
            azimuth_offsets_deg=(0.0, 45.0),
        )


def test_spherical_poses_custom_orbit_axis_constant_coordinate() -> None:
    """Orbit around +X: all eyes share the same X (pole direction) for fixed elevation."""
    poses = spherical_camera_poses(
        8,
        radius=1.0,
        elevation_deg=45.0,
        elevation_min_deg=25.0,
        elevation_max_deg=60.0,
        orbit_axis=np.array([1.0, 0.0, 0.0]),
    )
    xs = [float(p.eye[0]) for p in poses]
    assert all(abs(x - xs[0]) < 1e-6 for x in xs)


def test_rotate_vector_about_x_90_maps_y_to_z() -> None:
    v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    out = rotate_vector_about_axis(v, np.array([1.0, 0.0, 0.0]), 90.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 1.0], atol=1e-9)


def test_infer_pca_min_axis_mostly_perpendicular_to_elongation() -> None:
    pts = np.zeros((80, 3), dtype=np.float64)
    pts[:, 0] = np.linspace(-1.0, 1.0, 80)
    a = infer_orbit_axis_from_points(pts, mode="pca_min")
    assert abs(float(a[0])) < 0.35


def test_spherical_clamps_requested_elevation() -> None:
    poses = spherical_camera_poses(2, radius=1.0, elevation_deg=5.0, elevation_min_deg=15.0, elevation_max_deg=45.0)
    y = poses[0].eye[1]
    # Requested band [15,45] ∩ policy [25,60] → [25,45]; elevation 5° clamps to 25°.
    expect_y = math.sin(math.radians(25.0))
    assert abs(float(y) - expect_y) < 1e-6
