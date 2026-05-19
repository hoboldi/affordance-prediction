from pathlib import Path

import numpy as np
import pytest

from datasets.mesh_loading import load_mesh, normalize_mesh
from rendering.mesh_renderer import MeshRenderer

SAMPLE_MESH = Path(__file__).resolve().parents[1] / "data" / "sample.glb"


@pytest.mark.skipif(not SAMPLE_MESH.exists(), reason="sample.glb not present")
def test_mesh_renderer_produces_views():
    mesh = normalize_mesh(load_mesh(SAMPLE_MESH))
    renderer = MeshRenderer()
    views = renderer.render(mesh)

    assert len(views) == renderer.config.num_views
    for view in views:
        assert view.rgb.shape[:2] == (renderer.config.image_size, renderer.config.image_size)
        assert view.depth.shape == view.rgb.shape[:2]
        assert view.vertex_uv.shape == (mesh.num_vertices, 2)
        assert view.vertex_visible.shape == (mesh.num_vertices,)
        assert view.vertex_visible.sum() > 0
        assert view.rgb.mean() > 5  # not all black
