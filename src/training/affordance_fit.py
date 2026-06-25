"""Minimal supervised fit for :class:`models.mlp_head.AffordanceMLP` on per-vertex tensors."""

from __future__ import annotations

import torch

from models.mlp_head import AffordanceMLP, affordance_bce_loss


def fit_affordance_mlp_simple(
    model: AffordanceMLP,
    vlm_features: torch.Tensor,
    verb_embedding: torch.Tensor,
    labels: torch.Tensor,
    *,
    vertex_mask: torch.Tensor | None = None,
    sam3d_global: torch.Tensor | None = None,
    max_steps: int = 200,
    lr: float = 1e-4,
    subset_size: int | None = 500,
) -> list[float]:
    """
    Adam + BCE-with-logits on a (optional) subset of vertices.

    ``vlm_features`` / ``labels`` / ``vertex_mask`` must share the same leading size ``V``.
    If ``vertex_mask`` is set, vertices are taken in index order among ``True`` entries,
    truncated to ``subset_size`` when given.
    """
    if vlm_features.ndim != 2:
        raise ValueError(f"vlm_features must be (V, D), got shape {tuple(vlm_features.shape)}")
    vlm_features = vlm_features.detach()
    labels = labels.detach().to(vlm_features.dtype)
    verb_embedding = verb_embedding.detach().to(vlm_features.dtype)

    if vertex_mask is not None:
        idx = vertex_mask.nonzero(as_tuple=True)[0]
    else:
        idx = torch.arange(vlm_features.shape[0], device=vlm_features.device, dtype=torch.long)
    if subset_size is not None and idx.numel() > subset_size:
        idx = idx[:subset_size]

    feat_sub = vlm_features[idx]
    label_sub = labels[idx]

    if model.cfg.sam3d_dim > 0:
        if sam3d_global is None:
            raise ValueError("sam3d_global is required when model.cfg.sam3d_dim > 0")
        sam3d_global = sam3d_global.detach().to(vlm_features.dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(max_steps):
        optimizer.zero_grad()
        logits = model(feat_sub, verb_embedding, sam3d_global)
        loss = affordance_bce_loss(logits, label_sub)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses
