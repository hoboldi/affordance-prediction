import numpy as np
import pytest

from affordance.visualization.robot_demo import (
    GraspTarget,
    ObjectData,
    estimate_half_gap,
    heatmap_colors,
    normalized_scores,
    select_grasp,
)


def _sphere_object(n: int = 2000, radius: float = 0.05, seed: int = 0) -> ObjectData:
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    positions = normals * radius + np.array([0.5, 0.0, 0.45])
    return ObjectData(
        positions=positions,
        normals=normals,
        faces=np.zeros((1, 3), dtype=np.int64),
        rgb=np.full((n, 3), 0.5),
        scores=rng.random(n).astype(np.float32),
    )


def test_normalized_scores_range():
    s = normalized_scores(np.array([0.0, 0.1, 0.5, 1.0, 100.0]))
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_select_grasp_prefers_high_scores():
    obj = _sphere_object()
    obj.scores[:] = 0.0
    target_idx = np.argmax(obj.normals @ np.array([-1.0, -1.0, 0.2]))
    obj.scores[target_idx] = 1.0
    grasp = select_grasp(obj, facing_dir=np.array([-0.7, -0.7, 0.14]))
    assert np.allclose(grasp.point, obj.positions[target_idx])


def test_select_grasp_avoids_bottom_approach():
    obj = _sphere_object()
    # hottest vertex faces straight down; a reachable one is slightly cooler
    bottom = np.argmin(obj.normals[:, 2])
    side = np.argmax(obj.normals @ np.array([-1.0, 0.0, 0.0]))
    obj.scores[:] = 0.0
    obj.scores[bottom] = 1.0
    obj.scores[side] = 0.99
    grasp = select_grasp(obj, facing_dir=np.array([-1.0, 0.0, 0.0]))
    assert grasp.normal[2] >= -0.3


def test_grasp_rotation_is_orthonormal_with_approach_into_surface():
    obj = _sphere_object()
    grasp = select_grasp(obj, facing_dir=np.array([-1.0, 0.0, 0.0]))
    R = grasp.rotation
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.dot(R[:, 2], -grasp.normal) == pytest.approx(1.0)


def test_estimate_half_gap_matches_sphere_chord():
    obj = _sphere_object(n=20000, radius=0.05)
    grasp = select_grasp(obj, facing_dir=np.array([-1.0, 0.0, 0.0]))
    half_gap = estimate_half_gap(obj, grasp, radius=0.02)
    # near-tangential chord of a 5 cm sphere within a 2 cm ball is < 2 cm
    assert 0.004 <= half_gap <= 0.02


def test_heatmap_colors_highlights_hot_vertices():
    obj = _sphere_object()
    obj.scores[:] = 0.0
    obj.scores[0] = 1.0
    colors = heatmap_colors(obj)
    assert colors.shape == obj.rgb.shape
    # cold vertices keep the base appearance, the hot one turns hot-colormap red/yellow
    assert np.allclose(colors[1:], obj.rgb[1:], atol=0.1)
    assert colors[0, 0] > 0.8
