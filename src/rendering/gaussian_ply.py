"""
Load 3D Gaussian Splatting ``.ply`` exports (common *3DGS* layout: x,y,z, SH DC, opacity, scale, rot).

For **true** ellipsoid rasterisation use ``rendering.gaussian_gsplat_renderer`` (CUDA + ``gsplat``).
``rendering.gaussian_point_renderer`` draws Gaussian **centres** only (pyrender preview).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GaussianSplatPLY:
    """Arrays shaped ``(N, …)`` read from a 3DGS-style PLY."""

    means: np.ndarray  # (N, 3) float32
    f_dc: np.ndarray  # (N, 3) float32
    opacity: np.ndarray  # (N,) float32
    scales: np.ndarray  # (N, 3) float32
    rotations: np.ndarray  # (N, 4) float32
    rotation_layout: str
    property_names: tuple[str, ...]
    # Full Inria SH stack (``f_dc`` + ``f_rest_*``) for view-dependent colour in ``gsplat``:
    # shape ``(N, (sh_degree + 1) ** 2, 3)`` with ``sh_degree`` in ``0 … 3`` typically.
    sh_coeffs: np.ndarray | None = None
    sh_degree: int | None = None


def load_gaussian_splat_ply(path: str | Path) -> GaussianSplatPLY:
    """
    Load positions and appearance parameters from a 3DGS ``.ply``.

    Required properties: ``x, y, z``, ``f_dc_0``, ``f_dc_1``, ``f_dc_2``, ``opacity``,
    ``scale_0..2``, ``rot_0..3``.

    Optional ``f_rest_*`` (sorted by numeric suffix) are stacked with ``f_dc`` into
    ``(N, K, 3)`` with ``K = (sh_degree + 1) ** 2`` for ``gsplat``
    view-dependent shading. If absent, callers may still use degree-0 SH via ``f_dc`` only
    (see :func:`rendering.gaussian_gsplat_renderer.render_gaussian_splat_gsplat_views`).

    Non-DC layout is controlled by :func:`_f_rest_layout` (``AFFORDANCE_GSPLAT_F_REST_LAYOUT``).

    **Binary PLY:** every vertex property must use the same ``float`` width (common Inria export).
    """
    path = Path(path)
    data = path.read_bytes()
    m = re.search(br"end_header\r?\n", data)
    if not m:
        raise ValueError(f"Invalid PLY (missing end_header): {path}")
    header = data[: m.start()].decode("ascii", errors="replace")
    body = data[m.end() :]

    fmt = None
    n_verts = 0
    props: list[tuple[str, str]] = []
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element vertex "):
            n_verts = int(line.split()[-1])
        elif line.startswith("property "):
            parts = line.split()
            ptype, pname = parts[1], parts[2]
            props.append((pname, ptype))

    if fmt is None or n_verts <= 0:
        raise ValueError(f"Could not parse vertex count / format from PLY header: {path}")

    names = [p[0] for p in props]
    required = [
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
    for r in required:
        if r not in names:
            raise ValueError(
                f"PLY missing required property {r!r}. Found: {names}. "
                "Export a standard 3DGS PLY (Inria / Nerfstudio style)."
            )

    if fmt == "ascii":
        block = body.decode("ascii", errors="replace").strip().splitlines()
        if len(block) < n_verts:
            raise ValueError(f"Expected {n_verts} vertex lines, got {len(block)}")
        rows = []
        for i in range(n_verts):
            vals = [float(x) for x in block[i].split()]
            if len(vals) < len(names):
                raise ValueError(
                    f"Line {i} has {len(vals)} values, expected at least {len(names)}"
                )
            rows.append(vals[: len(names)])
        table = np.asarray(rows, dtype=np.float32)
    elif fmt == "binary_little_endian":
        for _, ptype in props:
            if ptype != "float":
                raise ValueError(
                    f"Binary PLY with non-float property {ptype!r} is not supported yet. "
                    "Convert to all-float 3DGS export or ASCII PLY."
                )
        stride = 4 * len(props)
        expected = stride * n_verts
        if len(body) < expected:
            raise ValueError(f"Truncated binary PLY body: need {expected} bytes, have {len(body)}")
        table = np.frombuffer(body[:expected], dtype="<f4").reshape(n_verts, len(props))
    else:
        raise ValueError(f"Unsupported PLY format {fmt!r} (use ascii or binary_little_endian)")

    idx = {n: j for j, n in enumerate(names)}
    means = table[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    f_dc = table[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]].astype(np.float32)
    opacity = table[:, idx["opacity"]].astype(np.float32)
    scales = table[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]].astype(np.float32)
    rotations = table[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]].astype(np.float32)

    sh_coeffs, sh_degree = _stack_sh_coefficients_from_ply(names, table, idx, f_dc, f_rest_layout=_f_rest_layout())

    return GaussianSplatPLY(
        means=means,
        f_dc=f_dc,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        rotation_layout="wxyz",
        property_names=tuple(names),
        sh_coeffs=sh_coeffs,
        sh_degree=sh_degree,
    )


def _f_rest_layout() -> str:
    """
    How ``f_rest_*`` columns map to ``(N, K, 3)`` SH bands (excluding DC).

    * ``inria`` — **channel-major** flattening (Inria / Nerfstudio ``sh_coeffs`` PLY export): all
      non-DC coefficients for R, then G, then B (see ``graphdeco-inria/gaussian-splatting``).
    * ``rgb_interleaved`` — for each non-DC SH index, ``R, G, B`` columns appear consecutively
      (some third-party tools); set ``AFFORDANCE_GSPLAT_F_REST_LAYOUT=rgb_interleaved`` if
      gsplat renders look gray / wrong while other viewers look fine.
    """
    raw = os.environ.get("AFFORDANCE_GSPLAT_F_REST_LAYOUT", "inria").strip().lower()
    if raw in ("inria", "rgb_interleaved"):
        return raw
    return "inria"


def _stack_sh_coefficients_from_ply(
    names: tuple[str, ...] | list[str],
    table: np.ndarray,
    idx: dict[str, int],
    f_dc: np.ndarray,
    *,
    f_rest_layout: str = "inria",
) -> tuple[np.ndarray | None, int | None]:
    """
    Build ``(N, K, 3)`` SH coefficients and ``sh_degree`` from Inria-style ``f_rest_*`` columns.

    Inria ``save_ply`` flattens ``transpose(features_rest)`` with ``features_rest`` of shape
    ``(N, K-1, 3)`` — i.e. **channel-major** in the flat file: all ``K-1`` non-DC coefficients
    for **R**, then for **G**, then for **B** (see ``graphdeco-inria/gaussian-splatting`` ``GaussianModel.save_ply``).
    Property ``f_rest_j`` with ``j = c * (K-1) + s`` maps to RGB channel ``c`` and SH index ``1+s``.

    With ``f_rest_layout="rgb_interleaved"``, ``j`` maps as ``SH slot = j // 3``, ``channel = j % 3``.
    """
    n = int(table.shape[0])
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda x: int(x.rsplit("_", 1)[-1]),
    )
    if not rest_names:
        return None, None
    n_cols = len(rest_names)
    if n_cols % 3 != 0:
        raise ValueError(
            f"Found {n_cols} f_rest_* properties; Inria layout uses 3×(K−1) columns (RGB × non-DC bands)."
        )
    m = n_cols // 3
    k = m + 1
    r = int(math.sqrt(k))
    if r * r != k:
        raise ValueError(
            f"Inferred total SH bands K={k} from f_rest columns, but K must be a perfect square "
            f"(K = (sh_degree+1)^2). Check f_rest_* layout in the PLY."
        )
    sh_degree = r - 1
    out = np.zeros((n, k, 3), dtype=np.float32)
    out[:, 0, :] = f_dc.astype(np.float32, copy=False)
    for j, pname in enumerate(rest_names):
        col = idx[pname]
        if f_rest_layout == "rgb_interleaved":
            sh_slot = j // 3
            channel = j % 3
        else:
            channel = j // m
            sh_slot = j % m
        out[:, 1 + sh_slot, channel] = table[:, col]
    return out, sh_degree


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """Map degree-0 SH coefficients to RGB in ``[0, 1]`` (view-independent)."""
    c0 = 0.28209479177387814
    rgb = 0.5 + c0 * f_dc.astype(np.float64)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
