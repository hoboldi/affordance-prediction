"""Pure-numpy tests for gsplat viewpoint selection helpers (no CUDA)."""

from __future__ import annotations

import numpy as np

from rendering.gsplat_viewpoint_selection import (
    _meta_json_safe,
    _cov_elongation,
    apply_scene_normalize,
    transform_cam_to_world,
    unproject_depth_zbuffer,
)


def test_meta_json_safe_nonfinite_to_null() -> None:
    out = _meta_json_safe({"x": float("nan"), "y": [1.0, float("inf")], "z": {"a": float("-inf")}})
    assert out["x"] is None
    assert out["y"] == [1.0, None]
    assert out["z"]["a"] is None


def test_unproject_center_ray() -> None:
    h, w = 65, 65
    K = np.array([[100.0, 0.0, (w - 1) / 2.0], [0.0, 100.0, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
    depth = np.zeros((h, w), dtype=np.float32)
    depth[h // 2, w // 2] = 2.0
    pts = unproject_depth_zbuffer(depth, K, stride=1)
    assert pts.shape == (1, 3)
    assert abs(float(pts[0, 2]) - 2.0) < 1e-5
    assert abs(float(pts[0, 0])) < 1e-4 and abs(float(pts[0, 1])) < 1e-4


def test_apply_scene_normalize_simple() -> None:
    gt_n = apply_scene_normalize(
        np.array([[3.0, 0.0, 0.0]], dtype=np.float64),
        center=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        scale=0.95,
    )
    np.testing.assert_allclose(gt_n[0], [1.9, 0.0, 0.0], rtol=1e-5, atol=1e-5)


def test_transform_cam_to_world_identity() -> None:
    c2w = np.eye(4, dtype=np.float64)
    p = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    w = transform_cam_to_world(p, c2w)
    np.testing.assert_allclose(w, [[0.0, 0.0, 1.0]], atol=1e-5)


def test_elongation_sphere_low() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3)).astype(np.float64) * 0.01
    assert _cov_elongation(pts) < 5.0
