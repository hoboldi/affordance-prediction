from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rendering.gaussian_ply import GaussianSplatPLY, load_gaussian_splat_ply, sh_dc_to_rgb, sigmoid
from rendering.gaussian_point_renderer import render_gaussian_splat_views, splat_ply_to_point_cloud
from rendering.mesh_renderer import MeshRenderConfig


def test_load_example_tiny_gaussians_ply() -> None:
    root = Path(__file__).resolve().parents[1]
    ply_path = root / "examples" / "gaussian_splat" / "tiny_gaussians.ply"
    g = load_gaussian_splat_ply(ply_path)
    assert g.means.shape == (400, 3)
    assert g.f_dc.shape == (400, 3)
    assert g.opacity.shape == (400,)
    assert np.isfinite(g.means).all()


def test_sh_dc_to_rgb_range() -> None:
    rgb = sh_dc_to_rgb(np.zeros((2, 3), dtype=np.float32))
    assert rgb.shape == (2, 3)
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_sigmoid() -> None:
    assert sigmoid(np.array([0.0], dtype=np.float32))[0] == pytest.approx(0.5, abs=1e-3)


def test_splat_ply_to_point_cloud_subsample() -> None:
    means = np.random.randn(1000, 3).astype(np.float32)
    f_dc = np.zeros((1000, 3), dtype=np.float32)
    op = np.full(1000, 3.0, dtype=np.float32)
    sc = np.full((1000, 3), -2.0, dtype=np.float32)
    rot = np.zeros((1000, 4), dtype=np.float32)
    rot[:, 0] = 1.0
    ply = GaussianSplatPLY(
        means=means,
        f_dc=f_dc,
        opacity=op,
        scales=sc,
        rotations=rot,
        rotation_layout="wxyz",
        property_names=tuple(),
    )
    p, c = splat_ply_to_point_cloud(ply, max_points=50, seed=1, normalize_scene=True)
    assert p.shape == (50, 3) and c.shape == (50, 3)


def test_render_gaussian_splat_views_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    ply_path = root / "examples" / "gaussian_splat" / "tiny_gaussians.ply"
    cfg = MeshRenderConfig(image_size=128, num_views=4, camera_radius=2.0)
    try:
        views = render_gaussian_splat_views(ply_path, cfg, max_points=200, seed=0)
    except Exception as exc:  # noqa: BLE001 — EGL / display varies by machine
        pytest.skip(f"pyrender unavailable in this environment: {exc}")
    assert len(views) == 4
    assert views[0].rgb.shape == (128, 128, 3)
    assert views[0].depth.shape == (128, 128)
