from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rendering.gaussian_gsplat_renderer import render_gaussian_splat_gsplat_views
from rendering.mesh_renderer import MeshRenderConfig


def test_render_gaussian_gsplat_views_smoke() -> None:
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for gsplat rasterization")

    root = Path(__file__).resolve().parents[1]
    ply_path = root / "examples" / "gaussian_splat" / "tiny_gaussians.ply"
    cfg = MeshRenderConfig(image_size=128, num_views=2, camera_radius=2.0)
    views = render_gaussian_splat_gsplat_views(
        ply_path,
        cfg,
        max_points=400,
        normalize_scene=True,
        seed=0,
    )
    assert len(views) == 2
    assert views[0].rgb.shape == (128, 128, 3)
    assert views[0].depth.shape == (128, 128)
    assert int(views[0].rgb.max()) > 5
