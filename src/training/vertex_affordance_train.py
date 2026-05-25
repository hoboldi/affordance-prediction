"""Per-mesh vertex affordance training (variable V — iterate samples, no fixed batching)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from models.mlp_head import AffordanceMLP, affordance_bce_loss


def _item_to_device(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(item)
    if out.get("vertex_features") is not None:
        out["vertex_features"] = out["vertex_features"].to(device)
    if out.get("vertex_visible_mask") is not None:
        out["vertex_visible_mask"] = out["vertex_visible_mask"].to(device)
    if out.get("vertex_affordance") is not None:
        out["vertex_affordance"] = out["vertex_affordance"].to(device)
    if out.get("sam3d_global_latent") is not None:
        out["sam3d_global_latent"] = out["sam3d_global_latent"].float().to(device)
    return out


def training_epoch_vertex_bce(
    model: AffordanceMLP,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    *,
    verb_embeddings: dict[str, torch.Tensor],
    device: torch.device,
    max_samples: int | None = None,
) -> float:
    """
    One full pass over ``dataset`` (each item = one mesh). Returns mean BCE loss per sample.
    """
    model.train()
    losses: list[float] = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for i in range(n):
        raw = dataset[i]
        if raw.get("vertex_features") is None:
            raise ValueError(
                f"sample {raw.get('sample_id')!r} missing vertex_features — add vertex_semantics_path to manifest"
            )
        if raw.get("vertex_affordance") is None:
            raise ValueError(f"sample {raw.get('sample_id')!r} missing vertex_affordance labels")
        item = _item_to_device(raw, device)
        verb = item["verb"]
        if verb not in verb_embeddings:
            raise KeyError(f"No embedding cached for verb {verb!r}; precompute verb_embeddings keys")
        verb_emb = verb_embeddings[verb].to(device)
        sam3d = item.get("sam3d_global_latent")
        if model.cfg.sam3d_dim > 0 and sam3d is None:
            raise ValueError(f"Model expects SAM3D latent but sample {item['sample_id']!r} has none")

        v_feat = item["vertex_features"]
        y = item["vertex_affordance"]
        if y.shape[0] != v_feat.shape[0]:
            raise ValueError(
                f"vertex / label length mismatch sample {item['sample_id']!r}: {v_feat.shape[0]} vs {y.shape[0]}"
            )

        optimizer.zero_grad()
        logits = model(
            v_feat,
            verb_emb,
            sam3d if model.cfg.sam3d_dim > 0 else None,
        )
        mask = item.get("vertex_visible_mask")
        loss = affordance_bce_loss(logits, y, mask)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def eval_vertex_bce(
    model: AffordanceMLP,
    dataset: Dataset,
    *,
    verb_embeddings: dict[str, torch.Tensor],
    device: torch.device,
    max_samples: int | None = None,
) -> float:
    """Mean BCE (eval mode) over ``dataset``."""
    model.eval()
    losses: list[float] = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for i in range(n):
        raw = dataset[i]
        if raw.get("vertex_features") is None or raw.get("vertex_affordance") is None:
            raise ValueError(f"sample {raw.get('sample_id')!r} missing features or labels")
        item = _item_to_device(raw, device)
        verb_emb = verb_embeddings[item["verb"]].to(device)
        sam3d = item.get("sam3d_global_latent")
        logits = model(
            item["vertex_features"],
            verb_emb,
            sam3d if model.cfg.sam3d_dim > 0 else None,
        )
        mask = item.get("vertex_visible_mask")
        loss = affordance_bce_loss(logits, item["vertex_affordance"], mask)
        losses.append(float(loss.cpu()))
    return sum(losses) / max(len(losses), 1)
