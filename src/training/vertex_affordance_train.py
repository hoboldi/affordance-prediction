"""Per-mesh vertex affordance training (variable V — iterate samples, no fixed batching)."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from models.mlp_head import AffordanceMLP, affordance_bce_loss


def _item_to_device(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(item)
    for key in (
        "vertex_features",
        "vertex_affordance",
        "slat_vertex_features",
        "vertex_normals",
        "dino_cls",
        "ss_dino_cls",
    ):
        if out.get(key) is not None:
            out[key] = out[key].float().to(device)
    # Keep mask as bool
    if out.get("vertex_visible_mask") is not None:
        out["vertex_visible_mask"] = out["vertex_visible_mask"].bool().to(device)
    return out


def _check_required_features(
    raw: dict[str, Any],
    cfg: Any,
    sample_id: str | None,
) -> None:
    checks = [
        ("vlm_dim", "vertex_features"),
        ("sam3d_dim", "slat_vertex_features"),
        ("dino_cls_dim", "dino_cls"),
        ("ss_dino_cls_dim", "ss_dino_cls"),
        ("normals_dim", "vertex_normals"),
    ]
    for dim_attr, key in checks:
        if getattr(cfg, dim_attr, 0) > 0 and raw.get(key) is None:
            raise ValueError(f"sample {sample_id!r} missing {key!r} ({dim_attr}={getattr(cfg, dim_attr)})")


def _balanced_vertex_sample(
    y: torch.Tensor,
    n_per_class: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return indices of up to n_per_class positives and n_per_class negatives.

    If mask is provided, only consider vertices where mask=True.
    Falls back to all available vertices of a class when fewer than n_per_class exist.
    """
    candidate = mask.nonzero(as_tuple=True)[0] if mask is not None else torch.arange(y.shape[0], device=y.device)
    pos_idx = candidate[y[candidate] > 0.5]
    neg_idx = candidate[y[candidate] <= 0.5]

    def _sample(idx: torch.Tensor, n: int) -> torch.Tensor:
        if idx.numel() <= n:
            return idx
        perm = torch.randperm(idx.numel(), device=idx.device)[:n]
        return idx[perm]

    return torch.cat([_sample(pos_idx, n_per_class), _sample(neg_idx, n_per_class)])


def _forward(
    model: AffordanceMLP,
    item: dict[str, Any],
    verb_idx: int | None,
) -> torch.Tensor:
    return model(
        verb_idx,
        slat_vertex=item.get("slat_vertex_features"),
        vlm_features=item.get("vertex_features"),
        dino_cls=item.get("dino_cls"),
        ss_dino_cls=item.get("ss_dino_cls"),
        vertex_normals=item.get("vertex_normals"),
    )


def training_epoch_vertex_bce(
    model: AffordanceMLP,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    *,
    verb_to_idx: dict[str, int],
    device: torch.device,
    max_samples: int | None = None,
    progress: Callable[[range], Any] | None = None,
    pos_weight: float = 5.0,
    n_vertices_per_class: int = 2048,
    grad_accum: int = 8,
) -> float:
    """One full pass over ``dataset`` (each item = one mesh). Returns mean BCE loss per sample.

    For each mesh, ``n_vertices_per_class`` positives and the same number of negatives are
    sampled to form a balanced minibatch. Gradients are accumulated over ``grad_accum``
    samples before each optimizer step — single-sample updates produce conflicting gradients
    on the shared weights and collapse the model to the constant-prediction solution.
    """
    model.train()
    losses: list[float] = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    indices = progress(range(n)) if progress is not None else range(n)

    optimizer.zero_grad()
    pending = 0
    for j, i in enumerate(indices):
        raw = dataset[i]
        if raw.get("vertex_affordance") is None:
            raise ValueError(f"sample {raw.get('sample_id')!r} missing vertex_affordance labels")
        _check_required_features(raw, model.cfg, raw.get("sample_id"))

        item = _item_to_device(raw, device)
        verb = item["verb"]
        if verb not in verb_to_idx:
            raise KeyError(f"No index for verb {verb!r}")

        logits = _forward(model, item, verb_to_idx[verb])
        y = item["vertex_affordance"]

        idx = _balanced_vertex_sample(y, n_vertices_per_class, item.get("vertex_visible_mask"))
        # pos_weight=1.0 — balanced sampling already equalises classes
        loss = affordance_bce_loss(logits[idx], y[idx], pos_weight=1.0)
        (loss / grad_accum).backward()
        pending += 1
        if pending == grad_accum:
            optimizer.step()
            optimizer.zero_grad()
            pending = 0
        losses.append(float(loss.detach().cpu()))

    if pending > 0:  # flush leftover gradients
        optimizer.step()
        optimizer.zero_grad()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def eval_vertex_bce(
    model: AffordanceMLP,
    dataset: Dataset,
    *,
    verb_to_idx: dict[str, int],
    device: torch.device,
    max_samples: int | None = None,
    progress: Callable[[range], Any] | None = None,
    pos_weight: float = 5.0,
) -> float:
    """Mean BCE (eval mode, all vertices) over ``dataset``."""
    model.eval()
    losses: list[float] = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    indices = progress(range(n)) if progress is not None else range(n)
    for i in indices:
        raw = dataset[i]
        if raw.get("vertex_affordance") is None:
            raise ValueError(f"sample {raw.get('sample_id')!r} missing vertex_affordance labels")
        _check_required_features(raw, model.cfg, raw.get("sample_id"))
        item = _item_to_device(raw, device)
        logits = _forward(model, item, verb_to_idx[item["verb"]])
        loss = affordance_bce_loss(
            logits, item["vertex_affordance"],
            mask=item.get("vertex_visible_mask"),
            pos_weight=pos_weight,
        )
        losses.append(float(loss.cpu()))
    return sum(losses) / max(len(losses), 1)
