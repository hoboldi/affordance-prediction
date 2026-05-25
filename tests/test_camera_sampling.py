from __future__ import annotations

import math

from rendering.camera_sampling import clamp_elevation_deg, spherical_camera_poses


def test_clamp_elevation_deg() -> None:
    assert clamp_elevation_deg(5.0) == 30.0
    assert clamp_elevation_deg(90.0) == 50.0
    assert clamp_elevation_deg(40.0) == 40.0


def test_spherical_poses_fixed_elevation_varying_azimuth() -> None:
    poses = spherical_camera_poses(
        6,
        radius=2.0,
        elevation_deg=40.0,
        elevation_min_deg=30.0,
        elevation_max_deg=50.0,
    )
    ys = [float(p.eye[1]) for p in poses]
    assert all(abs(y - ys[0]) < 1e-9 for y in ys), "all views share the same height (fixed pitch)"
    rings = [math.hypot(float(p.eye[0]), float(p.eye[2])) for p in poses]
    assert all(abs(r - rings[0]) < 1e-9 for r in rings), "same horizontal ring radius"


def test_spherical_clamps_requested_elevation() -> None:
    poses = spherical_camera_poses(2, radius=1.0, elevation_deg=5.0, elevation_min_deg=15.0, elevation_max_deg=45.0)
    y = poses[0].eye[1]
    # Requested band [15,45] ∩ policy [30,50] → [30,45]; elevation 5° clamps to 30°.
    expect_y = math.sin(math.radians(30.0))
    assert abs(float(y) - expect_y) < 1e-6
