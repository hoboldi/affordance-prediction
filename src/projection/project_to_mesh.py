from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from projection.multiview_fusion import fuse_patch_features
from projection.patch_vertex_mapping import render_uv_to_patch_index
from vlm.feature_cache import load_patch_features
from vlm.patch_extractor import PatchFeatures


@dataclass
class VertexSemanticFeatures:
    """VLM features aggregated on mesh vertices."""

    features: torch.Tensor  # (V, D)
    view_counts: torch.Tensor  # (V,) number of views that contributed
    visible_in_any_view: np.ndarray  # (V,) bool


@dataclass
class ViewProjectionInputs:
    vertex_uv: np.ndarray
    vertex_visible: np.ndarray
    patches: PatchFeatures
    render_height: int
    render_width: int
    camera_position: np.ndarray | None = None  # (3,) camera world pos — enables facing-weighted fusion


def project_views_to_vertices(
    views: list[ViewProjectionInputs],
    *,
    clip_image_size: int = 224,
    vertex_normals: np.ndarray | None = None,
    vertex_positions: np.ndarray | None = None,
) -> VertexSemanticFeatures:
    """Project patch tokens onto vertices and fuse across views.

    If ``vertex_normals`` + ``vertex_positions`` are given and every view carries a ``camera_position``,
    fusion is **facing-weighted**: each view contributes to a vertex in proportion to how head-on the
    surface faces that camera (``clip(normal·dir_to_cam, 0.1, 1)``). This down-weights grazing/
    foreshortened views and smooths the seams a plain equal-weight mean leaves. Otherwise: equal mean.
    """
    if not views:
        raise ValueError("views must not be empty")

    num_vertices = views[0].vertex_uv.shape[0]
    patch_indices_per_view: list[np.ndarray] = []
    patches_per_view: list[torch.Tensor] = []

    for view in views:
        pf = view.patches
        patch_indices_per_view.append(
            render_uv_to_patch_index(
                view.vertex_uv,
                view.vertex_visible,
                render_height=view.render_height,
                render_width=view.render_width,
                clip_image_size=clip_image_size,
                grid_h=pf.grid_h,
                grid_w=pf.grid_w,
            )
        )
        patches_per_view.append(pf.patches)

    view_weights: list[np.ndarray] | None = None
    if (
        vertex_normals is not None
        and vertex_positions is not None
        and all(v.camera_position is not None for v in views)
    ):
        n = np.asarray(vertex_normals, dtype=np.float64)
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
        pos = np.asarray(vertex_positions, dtype=np.float64)
        view_weights = []
        for view in views:
            d = np.asarray(view.camera_position, dtype=np.float64)[None, :] - pos
            d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
            facing = (n * d).sum(axis=1)
            view_weights.append(np.clip(facing, 0.1, 1.0).astype(np.float32))  # floor: visible verts always count

    features, counts = fuse_patch_features(
        num_vertices,
        patch_indices_per_view,
        patches_per_view,
        view_weights=view_weights,
    )

    visible_any = np.zeros(num_vertices, dtype=bool)
    for patch_idx in patch_indices_per_view:
        visible_any |= patch_idx >= 0

    return VertexSemanticFeatures(
        features=features,
        view_counts=counts,
        visible_in_any_view=visible_any,
    )


def load_cached_views(
    render_cache: Path,
    vlm_cache: Path,
) -> list[ViewProjectionInputs]:
    """Load paired render correspondences and patch features from notebook caches."""
    render_cache = Path(render_cache)
    vlm_cache = Path(vlm_cache)

    rgb_paths = sorted(render_cache.glob("rgb_*.npy"))
    if not rgb_paths:
        raise FileNotFoundError(f"No rgb_*.npy under {render_cache} — run notebook 02 first.")

    views: list[ViewProjectionInputs] = []
    for rgb_path in rgb_paths:
        stem = rgb_path.stem  # rgb_0
        idx = stem.split("_")[-1]
        uv_path = render_cache / f"vertex_uv_{idx}.npy"
        vis_path = render_cache / f"vertex_visible_{idx}.npy"
        patch_path = vlm_cache / f"patches_{idx}.pt"

        for p in (uv_path, vis_path, patch_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing cache file: {p}")

        rgb = np.load(rgb_path)
        views.append(
            ViewProjectionInputs(
                vertex_uv=np.load(uv_path),
                vertex_visible=np.load(vis_path),
                patches=load_patch_features(patch_path),
                render_height=int(rgb.shape[0]),
                render_width=int(rgb.shape[1]),
            )
        )

    return views


def vertex_verb_similarity(
    vertex_features: torch.Tensor,
    verb_embedding: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity per vertex to a single verb embedding (V,)."""
    if verb_embedding.ndim == 1:
        verb_embedding = verb_embedding.unsqueeze(0)
    v_norm = vertex_features / vertex_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    t_norm = verb_embedding / verb_embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (v_norm @ t_norm.T).squeeze(-1)


def vertex_pca_colors(vertex_features: torch.Tensor) -> np.ndarray:
    """RGB vertex colors (V, 3) uint8 from the first 3 PCA components."""
    x = vertex_features.numpy().astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = x @ vt[:3].T
    lo = components.min(axis=0, keepdims=True)
    hi = components.max(axis=0, keepdims=True)
    normalized = (components - lo) / (hi - lo + 1e-8)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)
