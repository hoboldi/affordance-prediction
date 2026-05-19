from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPHeadConfig:
    vlm_dim: int = 512
    verb_dim: int = 512
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.1


class AffordanceMLP(nn.Module):
    """Per-vertex affordance predictor: concat(VLM feature, verb embedding) → score."""

    def __init__(self, cfg: MLPHeadConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MLPHeadConfig()
        self.cfg = cfg
        in_dim = cfg.vlm_dim + cfg.verb_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, vlm_features: torch.Tensor, verb_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vlm_features:   (V, vlm_dim) — per-vertex VLM features
            verb_embedding: (verb_dim,) or (1, verb_dim) — L2-normalised verb vector

        Returns:
            scores: (V,) raw logits (apply sigmoid for probabilities)
        """
        if verb_embedding.ndim == 1:
            verb_embedding = verb_embedding.unsqueeze(0)
        verb_tiled = verb_embedding.expand(vlm_features.shape[0], -1)
        x = torch.cat([vlm_features, verb_tiled], dim=-1)
        return self.net(x).squeeze(-1)


def affordance_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy loss, optionally masked to visible vertices only.

    Args:
        logits:  (V,) raw scores from AffordanceMLP
        targets: (V,) float labels in [0, 1]
        mask:    (V,) bool — only compute loss where True (e.g. visible vertices)
    """
    if mask is not None:
        logits = logits[mask]
        targets = targets[mask]
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)
