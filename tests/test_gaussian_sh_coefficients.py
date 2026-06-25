"""Tests for stacking Inria-style ``f_rest_*`` into ``gsplat`` SH tensors."""

from __future__ import annotations

import numpy as np
import pytest

from rendering.gaussian_ply import _stack_sh_coefficients_from_ply


def _fake_ply_table(n: int, n_rest: int) -> tuple[tuple[str, ...], np.ndarray, dict[str, int], np.ndarray]:
    names = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    for i in range(n_rest):
        names.append(f"f_rest_{i}")
    idx = {name: j for j, name in enumerate(names)}
    table = np.random.randn(n, len(names)).astype(np.float32)
    f_dc = table[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]].copy()
    return tuple(names), table, idx, f_dc


def test_stack_sh_none_when_no_f_rest() -> None:
    names, table, idx, f_dc = _fake_ply_table(3, 0)
    c, d = _stack_sh_coefficients_from_ply(names, table, idx, f_dc)
    assert c is None and d is None


def test_stack_sh_degree3_from_45_f_rest() -> None:
    names, table, idx, f_dc = _fake_ply_table(4, 45)
    c, d = _stack_sh_coefficients_from_ply(names, table, idx, f_dc)
    assert d == 3
    assert c is not None
    assert c.shape == (4, 16, 3)
    np.testing.assert_array_equal(c[:, 0, :], f_dc)


def test_stack_sh_degree1_from_9_f_rest() -> None:
    """SH degree 1 → K=4 bands → 3×(K−1)=9 f_rest columns (Inria channel-major order)."""
    names, table, idx, f_dc = _fake_ply_table(2, 9)
    c, d = _stack_sh_coefficients_from_ply(names, table, idx, f_dc)
    assert d == 1
    assert c is not None
    assert c.shape == (2, 4, 3)


def test_stack_sh_rgb_interleaved_ordering() -> None:
    """``f_rest_j`` with ``j = 3 * sh_slot + channel`` (R,G,B per SH band after DC)."""
    k = 4
    m = k - 1
    n = 1
    names = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    for i in range(3 * m):
        names.append(f"f_rest_{i}")
    idx = {name: j for j, name in enumerate(names)}
    table = np.zeros((n, len(names)), dtype=np.float32)
    f_dc = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    table[0, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]] = f_dc[0]
    for j in range(3 * m):
        table[0, idx[f"f_rest_{j}"]] = float(100 + j)
    coeffs, deg = _stack_sh_coefficients_from_ply(
        tuple(names), table, idx, f_dc, f_rest_layout="rgb_interleaved"
    )
    assert deg == 1 and coeffs is not None
    np.testing.assert_array_equal(coeffs[0, 0, :], f_dc[0])
    for j in range(3 * m):
        sh_slot = j // 3
        channel = j % 3
        assert coeffs[0, 1 + sh_slot, channel] == pytest.approx(100.0 + j)


def test_stack_sh_inria_channel_major_ordering() -> None:
    """Property index j = c*(K-1) + s maps to channel c, SH slot 1+s (Inria save_ply flatten)."""
    k = 4
    m = k - 1
    n = 1
    names = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    for i in range(3 * m):
        names.append(f"f_rest_{i}")
    idx = {name: j for j, name in enumerate(names)}
    table = np.zeros((n, len(names)), dtype=np.float32)
    f_dc = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    table[0, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]] = f_dc
    for j in range(3 * m):
        table[0, idx[f"f_rest_{j}"]] = float(100 + j)
    coeffs, deg = _stack_sh_coefficients_from_ply(tuple(names), table, idx, f_dc)
    assert deg == 1 and coeffs is not None
    np.testing.assert_array_equal(coeffs[0, 0, :], f_dc[0])
    for j in range(3 * m):
        c = j // m
        s = j % m
        assert coeffs[0, 1 + s, c] == pytest.approx(100.0 + j)
