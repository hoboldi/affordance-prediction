from __future__ import annotations

import numpy as np
import torch


def fuse_patch_features(
    num_vertices: int,
    view_patch_indices: list[np.ndarray],
    view_patches: list[torch.Tensor],
    *,
    view_visible_masks: list[np.ndarray] | None = None,
    view_weights: list[np.ndarray] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted-mean-pool projected patch features across views.

    Args:
        num_vertices: V
        view_patch_indices: per view (V,) patch index or -1
        view_patches: per view (N_patches, D)
        view_visible_masks: optional per view (V,) bool; defaults to index >= 0
        view_weights: optional per view (V,) float per-vertex weight (e.g. surface-facing). When
            given, fusion is a weighted mean — down-weights grazing/foreshortened views, which is
            what smooths the seams an equal-weight mean leaves. None = uniform (1.0 per view).

    Returns:
        vertex_features (V, D), view_counts (V,) float (sum of weights when weighted)
    """
    if not view_patches:
        raise ValueError("need at least one view")

    feature_dim = view_patches[0].shape[-1]
    feat_sum = torch.zeros(num_vertices, feature_dim, dtype=torch.float32)
    counts = torch.zeros(num_vertices, dtype=torch.float32)

    for patch_idx, patches, visible_mask, weights in zip(
        view_patch_indices,
        view_patches,
        view_visible_masks or [None] * len(view_patches),
        view_weights or [None] * len(view_patches),
    ):
        if visible_mask is None:
            visible_mask = patch_idx >= 0

        if not np.any(visible_mask):
            continue

        vert_ids = np.where(visible_mask)[0]
        tok_ids = patch_idx[visible_mask]
        if weights is None:
            feat_sum[vert_ids] += patches[tok_ids]
            counts[vert_ids] += 1.0
        else:
            w = torch.as_tensor(np.asarray(weights)[vert_ids], dtype=torch.float32)
            feat_sum[vert_ids] += patches[tok_ids] * w.unsqueeze(-1)
            counts[vert_ids] += w

    floor = 1e-6 if view_weights is not None else 1.0
    counts_safe = counts.clamp(min=floor).unsqueeze(-1)
    vertex_features = feat_sum / counts_safe
    return vertex_features, counts
