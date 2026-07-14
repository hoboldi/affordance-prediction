"""Affordance prediction heads and losses."""

from models.gnn_head import AffordanceGNN, AffordanceGNNConfig, gnn_head_config_from_model_cfg
from models.mlp_head import (
    AffordanceMLP,
    MLPHeadConfig,
    affordance_bce_loss,
    build_affordance_mlp,
    mlp_head_config_from_model_cfg,
)

__all__ = [
    # ReVerb head (final model)
    "AffordanceGNN",
    "AffordanceGNNConfig",
    "gnn_head_config_from_model_cfg",
    # MLP baseline
    "AffordanceMLP",
    "MLPHeadConfig",
    "affordance_bce_loss",
    "build_affordance_mlp",
    "mlp_head_config_from_model_cfg",
]
