from __future__ import annotations

import torch

from models.mlp_head import (
    AffordanceMLP,
    MLPHeadConfig,
    affordance_bce_loss,
    build_affordance_mlp,
    mlp_head_config_from_model_cfg,
)


def test_mlp_head_config_from_model_cfg() -> None:
    cfg = mlp_head_config_from_model_cfg({"vlm_dim": 128, "verb_dim": 64, "sam3d_dim": 8, "hidden_dims": [32], "dropout": 0.2})
    assert cfg.input_dim == 128 + 64 + 8
    assert cfg.hidden_dims == (32,)


def test_affordance_mlp_forward_no_sam3d() -> None:
    m = AffordanceMLP(MLPHeadConfig(vlm_dim=16, verb_dim=8, num_verbs=3, sam3d_dim=0, hidden_dims=(32,), dropout=0.0))
    out = m(verb_idx=0, vlm_features=torch.randn(10, 16))
    assert out.shape == (10,)


def test_affordance_mlp_forward_with_sam3d() -> None:
    m = AffordanceMLP(MLPHeadConfig(vlm_dim=16, verb_dim=8, num_verbs=3, sam3d_dim=4, hidden_dims=(32,), dropout=0.0))
    out = m(verb_idx=1, vlm_features=torch.randn(10, 16), slat_vertex=torch.randn(10, 4))
    assert out.shape == (10,)


def test_build_affordance_mlp_include_sam3d_false() -> None:
    m = build_affordance_mlp(
        {"model": {"vlm_dim": 16, "verb_dim": 8, "num_verbs": 3, "sam3d_dim": 8, "hidden_dims": [32], "dropout": 0.0}},
        include_sam3d=False,
    )
    assert m.cfg.sam3d_dim == 0


def test_affordance_bce_masked() -> None:
    logits = torch.tensor([0.0, 10.0, -10.0])
    targets = torch.tensor([0.5, 1.0, 0.0])
    mask = torch.tensor([True, True, False])
    loss = affordance_bce_loss(logits, targets, mask)
    assert loss.ndim == 0
    assert 0.0 < float(loss) < 10.0
