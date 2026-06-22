from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class MLPHeadConfig:
    # Per-vertex (spatially-varying) inputs — carry discrimination
    vlm_dim: int = 512
    sam3d_dim: int = 0           # set to 8 when SAM3D SLAT vertex features are available
    normals_dim: int = 0         # set to 3 to include per-vertex surface normals
    pos_dim: int = 0             # set to 3 to include normalized per-vertex xyz position

    # Global (per-mesh constant) inputs — modulate the geometry stream via FiLM
    verb_dim: int = 512
    num_verbs: int = 0           # size of the learned verb embedding table (required if verb_dim > 0)
    dino_cls_dim: int = 0        # set to 1024 for SLAT-stage DINOv2 CLS token
    ss_dino_cls_dim: int = 0     # set to 1024 for SS-stage DINOv2 CLS token

    cond_dim: int = 128          # global (conditioning) encoder output dim
    hidden_dims: tuple[int, ...] = field(default_factory=lambda: (256, 128))  # FiLM-modulated geometry trunk
    dropout: float = 0.1
    # How the verb conditions the prediction:
    #   "concat" — verb embedding concatenated onto EVERY vertex's feature (per-vertex conditioning):
    #              the MLP predicts from (vertex_features, verb) jointly, so it can localize different
    #              regions for different verbs. (FiLM, being one (γ,β) per object, can only rescale the
    #              whole field — it shifts the global level per verb but cannot select regions.)
    #   "film"   — verb folded into the global FiLM stream (legacy; global level only).
    verb_conditioning: str = "concat"
    input_layernorm: bool = True  # LayerNorm the per-vertex input so the verb embedding doesn't drown the (unit-norm) CLIP features

    @property
    def _verb_in_vertex(self) -> bool:
        return self.verb_conditioning == "concat" and self.verb_dim > 0

    @property
    def per_vertex_dim(self) -> int:
        d = self.vlm_dim + self.sam3d_dim + self.normals_dim + self.pos_dim
        return d + (self.verb_dim if self._verb_in_vertex else 0)

    @property
    def global_dim(self) -> int:
        d = self.dino_cls_dim + self.ss_dino_cls_dim
        return d + (self.verb_dim if (self.verb_conditioning == "film" and self.verb_dim > 0) else 0)

    @property
    def input_dim(self) -> int:
        """Total raw feature dimensionality (per-vertex + global), for reporting."""
        return self.per_vertex_dim + self.global_dim


class AffordanceMLP(nn.Module):
    """Per-vertex affordance predictor with FiLM-modulated geometry encoder.

    Geometry trunk: [vlm | slat | normals] → MLP, where each hidden layer is FiLM-modulated
    (``h * (1 + γ) + β``) by parameters predicted from the global context. The global stream
    [dino_cls | ss_dino_cls | verb_emb] → encoder → (γ, β) lets the *same* geometry map to
    different affordances under different verbs/objects by re-routing which geometric features
    matter — not merely shifting the output.

    FiLM params are zero-initialised (identity modulation at start), so the geometry stream
    learns first; conditioning is layered in as training proceeds. Requires gradient
    accumulation across samples to avoid collapsing to the constant solution under the
    conflicting per-sample gradients of interleaved single-sample updates.

    Output: (V,) logits — apply sigmoid for probabilities.
    """

    def __init__(self, cfg: MLPHeadConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MLPHeadConfig()
        self.cfg = cfg

        if cfg.per_vertex_dim == 0:
            raise ValueError("At least one per-vertex input (vlm_dim, sam3d_dim, normals_dim) must be > 0")

        # Balance scales across the concatenated per-vertex input (CLIP unit-norm vs verb embedding vs slat/normals)
        self.in_norm = nn.LayerNorm(cfg.per_vertex_dim) if cfg.input_layernorm else None

        # Geometry trunk (per-vertex)
        self.geom_layers = nn.ModuleList()
        prev = cfg.per_vertex_dim
        for h in cfg.hidden_dims:
            self.geom_layers.append(nn.Linear(prev, h))
            prev = h
        self.act = nn.ReLU()
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(prev, 1)
        nn.init.constant_(self.out.bias, 0.0)  # balanced sampling → start at logit 0

        # Verb embedding — used per-vertex in "concat" mode, or in the global FiLM in "film" mode.
        if cfg.verb_dim > 0:
            if cfg.num_verbs <= 0:
                raise ValueError("num_verbs must be > 0 when verb_dim > 0")
            self.verb_emb = nn.Embedding(cfg.num_verbs, cfg.verb_dim)

        # Global (object-level) conditioning → FiLM params for every geometry layer
        self.use_global = cfg.global_dim > 0
        if self.use_global:
            self.global_encoder = nn.Sequential(
                nn.Linear(cfg.global_dim, cfg.cond_dim), nn.ReLU(),
                nn.Linear(cfg.cond_dim, cfg.cond_dim), nn.ReLU(),
            )
            self.film = nn.Linear(cfg.cond_dim, 2 * sum(cfg.hidden_dims))
            nn.init.zeros_(self.film.weight)   # identity modulation (γ=β=0) at init
            nn.init.zeros_(self.film.bias)

    def _per_vertex_input(
        self,
        verb_idx: int | torch.Tensor | None,
        slat_vertex: torch.Tensor | None,
        vlm_features: torch.Tensor | None,
        vertex_normals: torch.Tensor | None,
        vertex_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for t, dim, name in [
            (vlm_features, self.cfg.vlm_dim, "vlm_features"),
            (slat_vertex, self.cfg.sam3d_dim, "slat_vertex"),
            (vertex_normals, self.cfg.normals_dim, "vertex_normals"),
            (vertex_positions, self.cfg.pos_dim, "vertex_positions"),
        ]:
            if dim > 0:
                if t is None:
                    raise ValueError(f"{name} required (dim={dim})")
                if t.shape[-1] != dim:
                    raise ValueError(f"{name} last dim must be {dim}, got {tuple(t.shape)}")
                parts.append(t)
        feat = torch.cat(parts, dim=-1)  # (V, per-vertex feature dim)
        # "concat" conditioning: append the verb embedding to EVERY vertex, so the head predicts from
        # (vertex_features, verb) jointly and can localize different regions for different verbs.
        if self.cfg._verb_in_vertex:
            if verb_idx is None:
                raise ValueError("verb_idx required (verb_conditioning='concat')")
            idx = torch.as_tensor(verb_idx, device=self.verb_emb.weight.device, dtype=torch.long)
            v = self.verb_emb(idx).reshape(-1)  # (verb_dim,)
            feat = torch.cat([feat, v.unsqueeze(0).expand(feat.shape[0], -1)], dim=-1)
        return feat

    def _global_input(
        self,
        verb_idx: int | torch.Tensor | None,
        dino_cls: torch.Tensor | None,
        ss_dino_cls: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.cfg.dino_cls_dim > 0:
            if dino_cls is None:
                raise ValueError("dino_cls required (dim>0)")
            parts.append(dino_cls.flatten()[: self.cfg.dino_cls_dim])
        if self.cfg.ss_dino_cls_dim > 0:
            if ss_dino_cls is None:
                raise ValueError("ss_dino_cls required (dim>0)")
            parts.append(ss_dino_cls.flatten()[: self.cfg.ss_dino_cls_dim])
        if self.cfg.verb_dim > 0 and self.cfg.verb_conditioning == "film":
            if verb_idx is None:
                raise ValueError("verb_idx required (verb_dim>0)")
            idx = torch.as_tensor(verb_idx, device=self.verb_emb.weight.device, dtype=torch.long)
            parts.append(self.verb_emb(idx).flatten())
        return torch.cat(parts, dim=-1).unsqueeze(0)  # (1, global_dim)

    def forward(
        self,
        verb_idx: int | torch.Tensor | None = None,
        slat_vertex: torch.Tensor | None = None,
        vlm_features: torch.Tensor | None = None,
        dino_cls: torch.Tensor | None = None,
        ss_dino_cls: torch.Tensor | None = None,
        vertex_normals: torch.Tensor | None = None,
        vertex_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self._per_vertex_input(verb_idx, slat_vertex, vlm_features, vertex_normals, vertex_positions)  # (V, per_vertex_dim)
        if self.in_norm is not None:
            h = self.in_norm(h)

        film_params = None
        if self.use_global:
            g = self._global_input(verb_idx, dino_cls, ss_dino_cls)           # (1, global_dim)
            c = self.global_encoder(g)                                         # (1, cond_dim)
            film_params = self.film(c)                                         # (1, 2*sum(hidden))

        offset = 0
        for i, lin in enumerate(self.geom_layers):
            h = lin(h)
            if film_params is not None:
                H = self.cfg.hidden_dims[i]
                gamma = film_params[:, offset:offset + H]
                beta = film_params[:, offset + H:offset + 2 * H]
                offset += 2 * H
                h = h * (1.0 + gamma) + beta   # broadcast (1, H) over (V, H)
            h = self.drop(self.act(h))
        return self.out(h).squeeze(-1)


def affordance_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """Binary cross-entropy loss with positive-class upweighting.

    Args:
        logits:     (V,) raw scores from AffordanceMLP
        targets:    (V,) float labels in [0, 1]
        mask:       (V,) bool — only compute loss where True (e.g. visible vertices)
        pos_weight: scalar weight on positive examples. Use 1.0 with balanced vertex
                    sampling; use 5–10 when training on all vertices (~6% positive rate).
    """
    if mask is not None:
        logits = logits[mask]
        targets = targets[mask]
    pw = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def mlp_head_config_from_model_cfg(model_cfg: dict[str, Any]) -> MLPHeadConfig:
    """Build :class:`MLPHeadConfig` from a ``configs/*.yaml`` ``model:`` block."""
    hd = model_cfg.get("hidden_dims", [256, 128])
    return MLPHeadConfig(
        vlm_dim=int(model_cfg.get("vlm_dim", 512)),
        sam3d_dim=int(model_cfg.get("sam3d_dim", 0)),
        normals_dim=int(model_cfg.get("normals_dim", 0)),
        pos_dim=int(model_cfg.get("pos_dim", 0)),
        verb_dim=int(model_cfg.get("verb_dim", 512)),
        num_verbs=int(model_cfg.get("num_verbs", 0)),
        dino_cls_dim=int(model_cfg.get("dino_cls_dim", 0)),
        ss_dino_cls_dim=int(model_cfg.get("ss_dino_cls_dim", 0)),
        cond_dim=int(model_cfg.get("cond_dim", 128)),
        hidden_dims=tuple(int(x) for x in hd),
        dropout=float(model_cfg.get("dropout", 0.1)),
        verb_conditioning=str(model_cfg.get("verb_conditioning", "concat")),
        input_layernorm=bool(model_cfg.get("input_layernorm", True)),
    )


def build_affordance_mlp(
    cfg: dict[str, Any] | None = None,
    *,
    include_sam3d: bool | None = None,
) -> AffordanceMLP:
    """
    Instantiate :class:`AffordanceMLP` from merged project config (``model.*`` keys).

    If ``cfg`` is omitted, loads ``configs/default.yaml`` via :func:`utils.config.load_config`.

    ``include_sam3d``:
        * ``False`` — force ``sam3d_dim=0``.
        * ``True`` — ensure ``sam3d_dim`` is positive (default 8 if unset in cfg).
        * ``None`` — use ``model.sam3d_dim`` from YAML as-is.
    """
    if cfg is None:
        from utils.config import load_config

        cfg = load_config()
    m_raw = dict(cfg.get("model", {}))
    if include_sam3d is False:
        m_raw["sam3d_dim"] = 0
    elif include_sam3d is True:
        if int(m_raw.get("sam3d_dim", 0)) <= 0:
            m_raw["sam3d_dim"] = 8
    mcfg = mlp_head_config_from_model_cfg(m_raw)
    return AffordanceMLP(mcfg)
