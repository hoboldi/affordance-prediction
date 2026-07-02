"""Per-mesh vertex affordance training (variable V — iterate samples, no fixed batching)."""

from __future__ import annotations

import collections
import logging
import math
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from models.mlp_head import AffordanceMLP, affordance_bce_loss

log = logging.getLogger(__name__)


def _item_to_device(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(item)
    for key in (
        "vertex_features",
        "dino_vertex_features",
        "vertex_geom",
        "vertex_affordance",
        "slat_vertex_features",
        "vertex_normals",
        "vertex_positions",
        "dino_cls",
        "ss_dino_cls",
    ):
        if out.get(key) is not None:
            out[key] = out[key].float().to(device)
    # Keep mask as bool
    if out.get("vertex_visible_mask") is not None:
        out["vertex_visible_mask"] = out["vertex_visible_mask"].bool().to(device)
    # GNN kNN graph: keep as long (indices), move to device
    if out.get("vertex_knn") is not None:
        out["vertex_knn"] = out["vertex_knn"].long().to(device)
    return out


def _check_required_features(
    raw: dict[str, Any],
    cfg: Any,
    sample_id: str | None,
) -> None:
    checks = [
        ("vlm_dim", "vertex_features"),
        ("dino_vertex_dim", "dino_vertex_features"),
        ("geom_dim", "vertex_geom"),
        ("sam3d_dim", "slat_vertex_features"),
        ("dino_cls_dim", "dino_cls"),
        ("ss_dino_cls_dim", "ss_dino_cls"),
        ("normals_dim", "vertex_normals"),
        ("pos_dim", "vertex_positions"),
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
    kwargs: dict[str, Any] = dict(
        slat_vertex=item.get("slat_vertex_features"),
        vlm_features=item.get("vertex_features"),
        dino_cls=item.get("dino_cls"),
        ss_dino_cls=item.get("ss_dino_cls"),
        vertex_normals=item.get("vertex_normals"),
        vertex_positions=item.get("vertex_positions"),
        dino_vertex=item.get("dino_vertex_features"),
        vertex_geom=item.get("vertex_geom"),
    )
    # GNN backbone: thread the precomputed kNN graph (None → built on-the-fly from positions).
    # The MLP head does not accept knn_idx, so only pass it when the model is a GNN.
    # (_item_to_device has already moved vertex_knn to the right device as a long tensor.)
    if type(model).__name__ == "AffordanceGNN":
        kwargs["knn_idx"] = item.get("vertex_knn")
    return model(verb_idx, **kwargs)


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
    conf_weight: float = 0.0,
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
        loss = affordance_bce_loss(logits[idx], y[idx], pos_weight=1.0, conf_weight=conf_weight)
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


_PER_VERTEX_KEYS = (
    "vertex_features", "dino_vertex_features", "slat_vertex_features", "vertex_normals",
    "vertex_positions", "vertex_affordance", "vertex_visible_mask",
)


def _subsample_item(item: dict[str, Any], sub: torch.Tensor) -> dict[str, Any]:
    """Slice the per-vertex fields to `sub` (global dino_cls/ss_dino_cls left intact)."""
    out = dict(item)
    for key in _PER_VERTEX_KEYS:
        if out.get(key) is not None:
            out[key] = out[key][sub]
    return out


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-6)


def training_epoch_vertex_contrastive(
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
    contrastive_weight: float = 1.0,
    max_vertices: int = 60000,
    conf_weight: float = 0.0,
) -> float:
    """Like :func:`training_epoch_vertex_bce` but groups an object's verbs into one step and adds a
    contrastive term that penalizes cross-verb prediction CORRELATION on the same object. The GEAL
    labels are near-disjoint across verbs (corr ~-0.1, top-region IoU ~0), yet the per-row BCE lets the
    head collapse toward a verb-agnostic average; this term pushes predictions apart toward the labels'
    distinctness. Each object is subsampled to `max_vertices` to bound memory (multiple verbs held for
    the contrastive backward). `grad_accum` counts OBJECTS.
    """
    model.train()
    by_obj: dict[str, list[int]] = collections.OrderedDict()
    for i in range(len(dataset.rows)):
        by_obj.setdefault(str(dataset.rows[i].sam3d_reconstruction_dir), []).append(i)
    objs = list(by_obj.values())
    order = torch.randperm(len(objs)).tolist()
    if max_samples is not None:
        order = order[:max_samples]
    iterator = progress(range(len(order))) if progress is not None else range(len(order))

    optimizer.zero_grad()
    pending = 0
    losses: list[float] = []
    sims: list[float] = []
    for kk in iterator:
        row_idxs = objs[order[kk]]
        first = dataset[row_idxs[0]]
        if first.get("vertex_affordance") is None:
            continue
        v_total = first["vertex_affordance"].shape[0]
        sub = torch.randperm(v_total)[: min(v_total, max_vertices)]

        preds: list[torch.Tensor] = []
        bce_sum = None
        nv = 0
        for i in row_idxs:
            raw = dataset[i]
            if raw.get("vertex_affordance") is None:
                continue
            _check_required_features(raw, model.cfg, raw.get("sample_id"))
            verb = raw["verb"]
            if verb not in verb_to_idx:
                continue
            item = _item_to_device(_subsample_item(raw, sub), device)
            logits = _forward(model, item, verb_to_idx[verb])
            y = item["vertex_affordance"]
            idx = _balanced_vertex_sample(y, n_vertices_per_class, item.get("vertex_visible_mask"))
            bce = affordance_bce_loss(logits[idx], y[idx], pos_weight=1.0, conf_weight=conf_weight)
            bce_sum = bce if bce_sum is None else bce_sum + bce
            preds.append(torch.sigmoid(logits))
            nv += 1
        if nv == 0:
            continue

        loss = bce_sum / nv
        if contrastive_weight > 0 and len(preds) >= 2:
            # Hinge: penalize only POSITIVE cross-verb correlation (clamp at 0). A raw correlation
            # penalty has no floor and overshoots disjoint into strong anti-correlation (corr ~-0.7),
            # which distorts common verbs (contain crashes). The hinge pushes pairs to uncorrelated/
            # disjoint and then stops, matching the labels (corr ~-0.14, IoU 0) without over-separating.
            pair = [_pearson(preds[a], preds[b]).clamp(min=0.0) for a in range(len(preds)) for b in range(a + 1, len(preds))]
            csim = torch.stack(pair).mean()
            loss = loss + contrastive_weight * csim
            sims.append(float(csim.detach().cpu()))
        (loss / grad_accum).backward()
        pending += 1
        if pending >= grad_accum:
            optimizer.step()
            optimizer.zero_grad()
            pending = 0
        losses.append(float(loss.detach().cpu()))

    if pending > 0:
        optimizer.step()
        optimizer.zero_grad()
    if sims:
        log.info("  mean cross-verb pred corr (multi-verb objs) = %.3f", sum(sims) / len(sims))
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
