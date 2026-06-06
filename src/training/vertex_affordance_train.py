"""Per-mesh vertex affordance training (variable V — iterate samples, no fixed batching)."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from models.mlp_head import AffordanceMLP, affordance_bce_loss


def _item_to_device(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(item)
    for key in ("vertex_features", "vertex_visible_mask", "vertex_affordance"):
        if out.get(key) is not None:
            out[key] = out[key].to(device)
    if out.get("slat_vertex_features") is not None:
        out["slat_vertex_features"] = out["slat_vertex_features"].float().to(device)
    return out


def training_epoch_vertex_bce(
    model: AffordanceMLP,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    *,
    verb_embeddings: dict[str, torch.Tensor],
    device: torch.device,
    max_samples: int | None = None,
    progress: Callable[[range], Any] | None = None,
    pos_weight: float = 5.0,
) -> float:
    """
    One full pass over ``dataset`` (each item = one mesh). Returns mean BCE loss per sample.

    Pass ``progress=tqdm`` to wrap the inner loop with a progress bar.
    """
    model.train()
    losses: list[float] = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    indices = progress(range(n)) if progress is not None else range(n)
    for i in indices:
        raw = dataset[i]
        if raw.get("vertex_affordance") is None:
            raise ValueError(f"sample {raw.get('sample_id')!r} missing vertex_affordance labels")
        if model.cfg.vlm_dim > 0 and raw.get("vertex_features") is None:
            raise ValueError(f"sample {raw.get('sample_id')!r} missing vertex_features (vlm_dim={model.cfg.vlm_dim})")
        if model.cfg.sam3d_dim > 0 and raw.get("slat_vertex_features") is None:
            raise ValueError(f"sample {raw.get('sample_id')!r} missing slat_vertex_features (sam3d_dim={model.cfg.sam3d_dim})")

        item = _item_to_device(raw, device)
        verb = item["verb"]
        if verb not in verb_embeddings:
            raise KeyError(f"No embedding cached for verb {verb!r}")
        verb_emb = verb_embeddings[verb].to(device)

        optimizer.zero_grad()
        logits = model(
            verb_emb,
            slat_vertex=item.get("slat_vertex_features"),
            vlm_features=item.get("vertex_features"),
        )
        y = item["vertex_affordance"]
        loss = affordance_bce_loss(logits, y, item.get("vertex_visible_mask"), pos_weight=pos_weight)
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
    progress: Callable[[range], Any] | None = None,
    pos_weight: float = 5.0,
) -> float:
    """Mean BCE (eval mode) over ``dataset``."""
    model.eval()
    losses: list[float] = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    indices = progress(range(n)) if progress is not None else range(n)
    for i in indices:
        raw = dataset[i]
        if raw.get("vertex_affordance") is None:
            raise ValueError(f"sample {raw.get('sample_id')!r} missing vertex_affordance labels")
        item = _item_to_device(raw, device)
        verb_emb = verb_embeddings[item["verb"]].to(device)
        logits = model(
            verb_emb,
            slat_vertex=item.get("slat_vertex_features"),
            vlm_features=item.get("vertex_features"),
        )
        loss = affordance_bce_loss(logits, item["vertex_affordance"], item.get("vertex_visible_mask"), pos_weight=pos_weight)
        losses.append(float(loss.cpu()))
    return sum(losses) / max(len(losses), 1)
