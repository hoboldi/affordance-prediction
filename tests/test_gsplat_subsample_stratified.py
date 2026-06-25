"""Voxel-stratified gsplat subsampling covers sparse high-contrast geometry vs bulk surfaces."""

from __future__ import annotations

import numpy as np

from rendering.gaussian_gsplat_renderer import _subsample_ply_stratified, _subsample_ply_uniform
from rendering.gaussian_ply import GaussianSplatPLY


def _synthetic_ply_tight_blob_plus_outliers(*, n_bulk: int, n_out: int) -> GaussianSplatPLY:
    n = n_bulk + n_out
    means = np.zeros((n, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    means[:n_bulk] = rng.normal(size=(n_bulk, 3)).astype(np.float32) * 0.02
    means[n_bulk:, 0] = 80.0
    return GaussianSplatPLY(
        means=means,
        f_dc=np.zeros((n, 3), dtype=np.float32),
        opacity=np.zeros((n,), dtype=np.float32),
        scales=np.zeros((n, 3), dtype=np.float32),
        rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
        rotation_layout="wxyz",
        property_names=("x",),
    )


def test_stratified_subsample_reaches_distant_outliers() -> None:
    """Uniform draw often misses a tiny distant cluster; stratified keeps occupied voxels."""
    ply = _synthetic_ply_tight_blob_plus_outliers(n_bulk=9500, n_out=500)
    sub_s = _subsample_ply_stratified(ply, max_points=400, seed=42)
    assert float(sub_s.means[:, 0].max()) > 50.0, "stratified should include the x≈80 outlier cluster"

    sub_u = _subsample_ply_uniform(ply, max_points=400, seed=99)
    assert sub_u.means.shape[0] == 400
