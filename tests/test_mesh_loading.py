from pathlib import Path

import numpy as np
import pytest

from datasets.mesh_loading import load_mesh, normalize_mesh

SAMPLE_MESH = Path(__file__).resolve().parents[1] / "data" / "sample.glb"


@pytest.mark.skipif(not SAMPLE_MESH.exists(), reason="sample.glb not present")
def test_load_sample_glb():
    mesh = load_mesh(SAMPLE_MESH)
    assert mesh.vertices.ndim == 2 and mesh.vertices.shape[1] == 3
    assert mesh.faces.ndim == 2 and mesh.faces.shape[1] == 3
    assert mesh.num_vertices > 0
    assert mesh.num_faces > 0


@pytest.mark.skipif(not SAMPLE_MESH.exists(), reason="sample.glb not present")
def test_normalize_mesh_unit_extent():
    mesh = load_mesh(SAMPLE_MESH)
    normalized = normalize_mesh(mesh, center=True, unit_scale=True)
    radius = np.linalg.norm(normalized.vertices, axis=1).max()
    assert radius == pytest.approx(1.0, rel=1e-4, abs=1e-4)
    assert np.allclose(normalized.vertices.mean(axis=0), 0.0, atol=1e-5)
