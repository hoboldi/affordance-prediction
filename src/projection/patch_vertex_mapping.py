from __future__ import annotations

import numpy as np


def render_uv_to_patch_index(
    vertex_uv: np.ndarray,
    vertex_visible: np.ndarray,
    *,
    render_height: int,
    render_width: int,
    clip_image_size: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """
    Map per-vertex render pixel coordinates to flat CLIP patch indices.

    Args:
        vertex_uv: (V, 2) with x=u, y=v in render pixels; nan if unprojected.
        vertex_visible: (V,) bool.
        render_height, render_width: rasterized view size (e.g. 512).
        clip_image_size: CLIP input side length after processor resize (e.g. 224).
        grid_h, grid_w: patch grid (e.g. 7×7).

    Returns:
        (V,) int64 patch index, or -1 where not visible.
    """
    patch_size = clip_image_size / grid_h
    indices = np.full(vertex_uv.shape[0], -1, dtype=np.int64)

    if not np.any(vertex_visible):
        return indices

    u = vertex_uv[vertex_visible, 0].astype(np.float64)
    v = vertex_uv[vertex_visible, 1].astype(np.float64)

    u_clip = (u / render_width) * clip_image_size
    v_clip = (v / render_height) * clip_image_size

    col = np.floor(u_clip / patch_size).astype(np.int64)
    row = np.floor(v_clip / patch_size).astype(np.int64)
    col = np.clip(col, 0, grid_w - 1)
    row = np.clip(row, 0, grid_h - 1)

    flat = row * grid_w + col
    indices[vertex_visible] = flat
    return indices
