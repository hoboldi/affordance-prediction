from __future__ import annotations

import numpy as np
import trimesh

from reconstruction.mesh_utils import apply_vertex_world_rotation_to_sam3d_result


class _DummyResult:
    def __init__(self, mesh: trimesh.Trimesh | None) -> None:
        self.mesh_scene = mesh
        self.gaussian_splat = None
        self.shape_latent = np.zeros(1)
        self.slat_feats = np.zeros(1)
        self.slat_coords = np.zeros((1, 3))


def test_apply_vertex_world_rotation_rotates_mesh_vertices() -> None:
    mesh = trimesh.creation.box(extents=(0.2, 0.4, 0.6))
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    r = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    before = np.asarray(mesh.vertices, dtype=np.float64).copy()
    res = _DummyResult(mesh)
    apply_vertex_world_rotation_to_sam3d_result(res, r)
    after = np.asarray(mesh.vertices, dtype=np.float64)
    expected = (r @ before.T).T
    assert np.linalg.norm(after - expected) < 1e-5


def test_apply_vertex_world_rotation_noop_when_r_none() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    before = np.asarray(mesh.vertices, dtype=np.float64).copy()
    res = _DummyResult(mesh)
    apply_vertex_world_rotation_to_sam3d_result(res, None)
    assert np.allclose(np.asarray(mesh.vertices), before)
