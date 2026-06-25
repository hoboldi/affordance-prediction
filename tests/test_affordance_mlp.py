from __future__ import annotations

import torch

from models.mlp_head import (
    AffordanceMLP,
    MLPHeadConfig,
    affordance_bce_loss,
    build_affordance_mlp,
    mlp_head_config_from_model_cfg,
)
from training.affordance_fit import fit_affordance_mlp_simple


def test_mlp_head_config_from_model_cfg() -> None:
    cfg = mlp_head_config_from_model_cfg({"vlm_dim": 128, "verb_dim": 64, "sam3d_dim": 8, "hidden_dims": [32], "dropout": 0.2})
    assert cfg.input_dim == 128 + 64 + 8
    assert cfg.hidden_dims == (32,)


def test_affordance_mlp_forward_no_sam3d() -> None:
    m = AffordanceMLP(MLPHeadConfig(vlm_dim=16, verb_dim=8, sam3d_dim=0, hidden_dims=(32,), dropout=0.0))
    v = torch.randn(10, 16)
    t = torch.randn(8)
    out = m(v, t)
    assert out.shape == (10,)


def test_affordance_mlp_forward_with_sam3d() -> None:
    m = AffordanceMLP(MLPHeadConfig(vlm_dim=16, verb_dim=8, sam3d_dim=4, hidden_dims=(32,), dropout=0.0))
    v = torch.randn(10, 16)
    t = torch.randn(8)
    g = torch.randn(4)
    out = m(v, t, g)
    assert out.shape == (10,)


def test_build_affordance_mlp_include_sam3d_false() -> None:
    m = build_affordance_mlp({"model": {"vlm_dim": 16, "verb_dim": 8, "sam3d_dim": 8, "hidden_dims": [32], "dropout": 0.0}}, include_sam3d=False)
    assert m.cfg.sam3d_dim == 0


def test_affordance_bce_masked() -> None:
    logits = torch.tensor([0.0, 10.0, -10.0])
    targets = torch.tensor([0.5, 1.0, 0.0])
    mask = torch.tensor([True, True, False])
    loss = affordance_bce_loss(logits, targets, mask)
    assert loss.ndim == 0
    assert 0.0 < float(loss) < 10.0


def test_fit_affordance_mlp_simple_decreases_loss() -> None:
    torch.manual_seed(0)
    model = AffordanceMLP(MLPHeadConfig(vlm_dim=8, verb_dim=4, sam3d_dim=0, hidden_dims=(16,), dropout=0.0))
    v = torch.randn(64, 8)
    verb = torch.randn(4)
    labels = torch.sigmoid(v @ torch.randn(8))  # synthetic smooth labels
    mask = torch.ones(64, dtype=torch.bool)
    losses = fit_affordance_mlp_simple(model, v, verb, labels, vertex_mask=mask, max_steps=80, lr=0.02, subset_size=64)
    assert losses[-1] < losses[0]


def test_fit_with_sam3d_global() -> None:
    model = AffordanceMLP(MLPHeadConfig(vlm_dim=8, verb_dim=4, sam3d_dim=2, hidden_dims=(16,), dropout=0.0))
    v = torch.randn(32, 8)
    verb = torch.randn(4)
    g = torch.randn(2)
    labels = torch.rand(32)
    mask = torch.ones(32, dtype=torch.bool)
    losses = fit_affordance_mlp_simple(
        model, v, verb, labels, vertex_mask=mask, sam3d_global=g, max_steps=40, lr=0.05, subset_size=32
    )
    assert len(losses) == 40
