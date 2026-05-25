"""
Load 3D Gaussian Splatting ``.ply`` exports (common *3DGS* layout: x,y,z, SH DC, opacity, scale, rot).

For **true** ellipsoid rasterisation use ``rendering.gaussian_gsplat_renderer`` (CUDA + ``gsplat``).
``rendering.gaussian_point_renderer`` draws Gaussian **centres** only (pyrender preview).
"""

from __future__ import annotations

import re
import struct
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


def load_gaussian_splat_ply(path: str | Path) -> GaussianSplatPLY:
    """
    Load positions and appearance parameters from a 3DGS ``.ply``.

    Required properties: ``x, y, z``, ``f_dc_0``, ``f_dc_1``, ``f_dc_2``, ``opacity``,
    ``scale_0..2``, ``rot_0..3``. Other properties (``f_rest_*``, normals, …) are ignored
    if present after those columns in binary mode — **only supported when every property
    uses the same ``float`` binary width** (the common Inria / Nerfstudio export).
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
        raw = body[:expected]
        pack = "<" + "f" * len(props)
        table = np.zeros((n_verts, len(names)), dtype=np.float32)
        for i in range(n_verts):
            chunk = raw[i * stride : (i + 1) * stride]
            table[i] = struct.unpack(pack, chunk)
    else:
        raise ValueError(f"Unsupported PLY format {fmt!r} (use ascii or binary_little_endian)")

    idx = {n: j for j, n in enumerate(names)}
    means = table[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    f_dc = table[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]].astype(np.float32)
    opacity = table[:, idx["opacity"]].astype(np.float32)
    scales = table[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]].astype(np.float32)
    rotations = table[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]].astype(np.float32)

    return GaussianSplatPLY(
        means=means,
        f_dc=f_dc,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        rotation_layout="wxyz",
        property_names=tuple(names),
    )


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """Map degree-0 SH coefficients to RGB in ``[0, 1]`` (view-independent)."""
    c0 = 0.28209479177387814
    rgb = 0.5 + c0 * f_dc.astype(np.float64)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
