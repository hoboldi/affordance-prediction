from __future__ import annotations

import numpy as np

from rendering.gaussian_gsplat_renderer import _c2w_pyrender_gl_to_gsplat_opencv


def test_gl_to_opencv_preserves_camera_center() -> None:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = [1.0, 2.0, 3.0]
    out = _c2w_pyrender_gl_to_gsplat_opencv(m)
    assert np.allclose(out[:3, 3], [1.0, 2.0, 3.0])


def test_gl_to_opencv_rotation_is_orthogonal() -> None:
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = q
    out = _c2w_pyrender_gl_to_gsplat_opencv(m)
    r = out[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(r) - 1.0) < 1e-6
