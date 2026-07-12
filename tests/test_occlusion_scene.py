from pathlib import Path

import numpy as np

from affordance.visualization.occlusion_scene import (
    ObjectData,
    SceneObject,
    _camera_xyaxes,
    _euler_matrix,
    _frame_camera,
    _place,
)


def _box_object(scale: float = 1.0) -> ObjectData:
    # unit cube corners, arbitrary scores/rgb/normals
    corners = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, -1).T.astype(float) * scale
    n = len(corners)
    return ObjectData(
        positions=corners,
        normals=np.tile([0.0, 0.0, 1.0], (n, 1)),
        faces=np.zeros((1, 3), dtype=np.int64),
        rgb=np.full((n, 3), 0.5),
        scores=np.linspace(0, 1, n).astype(np.float32),
    )


def test_euler_matrix_is_a_rotation():
    R = _euler_matrix((37.0, -12.0, 210.0))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_euler_z_rotates_handle_direction():
    # +y should map to -x under a +90 deg yaw about z
    R = _euler_matrix((0.0, 0.0, 90.0))
    assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), atol=1e-9)


def test_place_scales_to_size_and_sits_on_table():
    obj = _box_object(scale=3.0)
    spec = SceneObject(
        reconstruction_dir=".", scores_path=".", name="x",
        size=0.2, position=(0.5, -0.1, 0.0),
    )
    _place(obj, spec)
    extent = obj.positions.max(0) - obj.positions.min(0)
    assert np.isclose(extent.max(), 0.2)  # largest dimension matches requested size
    assert np.isclose(obj.positions[:, 2].min(), 0.0)  # bottom rests on the table
    centre_xy = (obj.positions.min(0) + obj.positions.max(0))[:2] / 2
    assert np.allclose(centre_xy, [0.5, -0.1])  # centred on the requested xy


def test_label_name_from_category_and_verb():
    spec = SceneObject(
        reconstruction_dir=Path("data/reconstructions/laptop__112_13277_23636__v000"),
        scores_path=".", name="front", size=0.3, position=(0, 0, 0), verb="press",
    )
    assert spec.label_name == "laptop_press"


def test_frame_camera_targets_centre_at_expected_distance():
    center = np.array([0.1, 0.2, 0.15])
    cam_pos, cam_target = _frame_camera(center, size=0.2)
    assert np.allclose(cam_target, center)
    assert np.isclose(np.linalg.norm(cam_pos - center), 2.3 * 0.2)
    assert cam_pos[1] < center[1]  # camera sits on the -y (front) side


def test_camera_xyaxes_orthonormal_and_level():
    pos = np.array([0.0, -0.75, 0.42])
    target = np.array([0.0, 0.05, 0.11])
    vals = np.array([float(v) for v in _camera_xyaxes(pos, target).split()])
    right, up = vals[:3], vals[3:]
    assert np.isclose(np.linalg.norm(right), 1.0) and np.isclose(np.linalg.norm(up), 1.0)
    assert np.isclose(np.dot(right, up), 0.0, atol=1e-9)
    assert np.isclose(right[2], 0.0, atol=1e-9)  # horizon stays level (right vector has no z)
