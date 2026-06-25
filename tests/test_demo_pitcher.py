from __future__ import annotations

import numpy as np
import pytest

from rendering.demo_pitcher import build_demo_pitcher_mesh
from rendering.mesh_renderer import MeshRenderConfig, MeshRenderer


def test_build_demo_pitcher_mesh_non_degenerate() -> None:
    m = build_demo_pitcher_mesh()
    assert m.num_vertices > 200
    assert m.num_faces > 200
    assert m.vertex_colors is not None
    lo, hi = m.bounds()
    assert np.all(lo < hi)


def test_render_demo_pitcher_smoke() -> None:
    mesh = build_demo_pitcher_mesh()
    cfg = MeshRenderConfig(image_size=128, num_views=3, camera_radius=2.0)
    renderer = MeshRenderer(cfg)
    try:
        views = renderer.render(mesh)
    except Exception as exc:  # noqa: BLE001 — EGL / display varies by machine
        pytest.skip(f"pyrender unavailable in this environment: {exc}")
    assert len(views) == 3
    assert views[0].rgb.shape == (128, 128, 3)
    assert int(views[0].rgb.max()) > 10
