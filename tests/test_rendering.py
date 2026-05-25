from pathlib import Path

import numpy as np
import pytest

from datasets.mesh_loading import load_mesh, normalize_mesh
from rendering.mesh_renderer import MeshRenderer
from rendering.renderer import render_mesh_views

SAMPLE_MESH = Path(__file__).resolve().parents[1] / "data" / "sample.glb"
TINY_SPLAT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "gaussian_splat"
    / "tiny_gaussians.ply"
)


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
        assert view.normal_rgb is not None and view.normal_rgb.shape == view.rgb.shape
        assert view.depth_vis_rgb is not None and view.depth_vis_rgb.shape == view.rgb.shape


@pytest.mark.skipif(not SAMPLE_MESH.exists(), reason="sample.glb not present")
def test_mesh_renderer_skips_geometry_aux_when_disabled():
    mesh = normalize_mesh(load_mesh(SAMPLE_MESH))
    renderer = MeshRenderer(MeshRenderConfig(image_size=64, num_views=2, render_geometry_aux=False))
    views = renderer.render(mesh)
    assert views[0].normal_rgb is None and views[0].depth_vis_rgb is None


@pytest.mark.skipif(not SAMPLE_MESH.exists(), reason="sample.glb not present")
@pytest.mark.skipif(not TINY_SPLAT.exists(), reason="tiny_gaussians.ply not present")
def test_render_mesh_views_gaussian_keeps_mesh_correspondences():
    mesh = normalize_mesh(load_mesh(SAMPLE_MESH))
    base = {
        "rendering": {
            "num_views": 4,
            "image_size": 128,
            "backend": "mesh",
            "mesh_path": str(SAMPLE_MESH),
            "splat_path": None,
        }
    }
    mesh_only = render_mesh_views(mesh, base)
    hybrid_cfg = {
        "rendering": {
            **base["rendering"],
            "backend": "gaussian",
            "splat_path": str(TINY_SPLAT),
        }
    }
    hybrid = render_mesh_views(mesh, hybrid_cfg)
    assert len(hybrid) == len(mesh_only)
    for a, b in zip(hybrid, mesh_only, strict=True):
        assert a.vertex_uv.shape == b.vertex_uv.shape
        assert np.allclose(a.vertex_uv, b.vertex_uv, equal_nan=True)
        assert np.allclose(a.depth, b.depth)
        assert a.rgb.shape == b.rgb.shape
        assert a.normal_rgb is not None and b.normal_rgb is not None
        assert np.array_equal(a.normal_rgb, b.normal_rgb)
        assert a.depth_vis_rgb is not None and b.depth_vis_rgb is not None
        assert np.array_equal(a.depth_vis_rgb, b.depth_vis_rgb)


@pytest.mark.skipif(not SAMPLE_MESH.exists(), reason="sample.glb not present")
def test_render_mesh_views_gaussian_without_splat_falls_back_to_mesh():
    mesh = normalize_mesh(load_mesh(SAMPLE_MESH))
    cfg = {
        "rendering": {
            "num_views": 2,
            "image_size": 64,
            "backend": "gaussian",
            "mesh_path": str(SAMPLE_MESH),
            "splat_path": None,
        }
    }
    with pytest.warns(UserWarning, match="splat_path"):
        views = render_mesh_views(mesh, cfg)
    assert len(views) == 2
    assert views[0].vertex_uv.shape == (mesh.num_vertices, 2)
