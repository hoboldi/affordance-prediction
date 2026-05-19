from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPHeadConfig:
    vlm_dim: int = 512
    verb_dim: int = 512
    sam3d_dim: int = 0          # set to 8 when SAM3D latents are available; 0 = not used
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.1

    @property
    def input_dim(self) -> int:
        return self.vlm_dim + self.verb_dim + self.sam3d_dim


class AffordanceMLP(nn.Module):
    """Per-vertex affordance predictor.

    Input per vertex: concat(vlm_feature, verb_embedding[, sam3d_global])
    Output: (V,) logits — apply sigmoid for probabilities.

    sam3d_global is a single (sam3d_dim,) vector broadcast to all vertices,
    representing the global geometry of the reconstructed object.
    """

    def __init__(self, cfg: MLPHeadConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MLPHeadConfig()
        self.cfg = cfg

        layers: list[nn.Module] = []
        prev = cfg.input_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        vlm_features: torch.Tensor,
        verb_embedding: torch.Tensor,
        sam3d_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            vlm_features:   (V, vlm_dim)
            verb_embedding: (verb_dim,) or (1, verb_dim) — L2-normalised
            sam3d_global:   (sam3d_dim,) or (1, sam3d_dim) — global shape latent,
                            broadcast to all vertices. Required when cfg.sam3d_dim > 0.
        """
        V = vlm_features.shape[0]

        if verb_embedding.ndim == 1:
            verb_embedding = verb_embedding.unsqueeze(0)
        parts = [vlm_features, verb_embedding.expand(V, -1)]

        if self.cfg.sam3d_dim > 0:
            if sam3d_global is None:
                raise ValueError("sam3d_global required when cfg.sam3d_dim > 0")
            if sam3d_global.ndim == 1:
                sam3d_global = sam3d_global.unsqueeze(0)
            parts.append(sam3d_global.expand(V, -1))

        x = torch.cat(parts, dim=-1)
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
