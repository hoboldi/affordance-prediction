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
    dino_vertex_dim: int = 0     # set to 128 for per-vertex DINOv2 patch features (alongside CLIP vlm)
    dino_filename: str = "vertex_dino.pt"

    # Global (per-mesh constant) inputs — modulate the geometry stream via FiLM
    verb_dim: int = 512
    num_verbs: int = 0           # size of the learned verb embedding table (required if verb_dim > 0)
    dino_cls_dim: int = 0        # set to 1024 for SLAT-stage DINOv2 CLS token
    ss_dino_cls_dim: int = 0     # set to 1024 for SS-stage DINOv2 CLS token

    cond_dim: int = 128          # global (conditioning) encoder output dim
    hidden_dims: tuple[int, ...] = field(default_factory=lambda: (256, 128))
    dropout: float = 0.1
    verb_conditioning: str = "concat"
    input_layernorm: bool = True
    verb_embedding: str = "learned"
    verb_text_dim: int = 512
    verb_proj_deep: bool = False

    @property
    def _verb_in_vertex(self) -> bool:
        return self.verb_conditioning == "concat" and self.verb_dim > 0

    @property
    def per_vertex_dim(self) -> int:
        d = self.vlm_dim + self.sam3d_dim + self.normals_dim + self.pos_dim + self.dino_vertex_dim
        return d + (self.verb_dim if self._verb_in_vertex else 0)

    @property
    def global_dim(self) -> int:
        d = self.dino_cls_dim + self.ss_dino_cls_dim
        return d + (self.verb_dim if (self.verb_conditioning == "film" and self.verb_dim > 0) else 0)

    @property
    def input_dim(self) -> int:
        return self.per_vertex_dim + self.global_dim


class AffordanceMLP(nn.Module):
    """Per-vertex affordance predictor with FiLM-modulated geometry encoder.

    Geometry trunk: [vlm | slat | normals] → MLP, where each hidden layer is FiLM-modulated
    (h * (1 + γ) + β) by parameters predicted from the global context. The global stream
    [dino_cls | ss_dino_cls | verb_emb] → encoder → (γ, β) lets the same geometry map to
    different affordances under different verbs/objects.

    FiLM params are zero-initialised (identity modulation at start).
    Output: (V,) logits — apply sigmoid for probabilities.
    """

    def __init__(self, cfg: MLPHeadConfig | None = None, verb_text_embeddings: torch.Tensor | None = None) -> None:
        super().__init__()
        cfg = cfg or MLPHeadConfig()
        self.cfg = cfg

        if cfg.per_vertex_dim == 0:
            raise ValueError("At least one per-vertex input must be > 0")

        self.in_norm = nn.LayerNorm(cfg.per_vertex_dim) if cfg.input_layernorm else None

        self.geom_layers = nn.ModuleList()
        prev = cfg.per_vertex_dim
        for h in cfg.hidden_dims:
            self.geom_layers.append(nn.Linear(prev, h))
            prev = h
        self.act = nn.ReLU()
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(prev, 1)
        nn.init.constant_(self.out.bias, 0.0)

        if cfg.verb_dim > 0:
            if cfg.num_verbs <= 0:
                raise ValueError("num_verbs must be > 0 when verb_dim > 0")
            if cfg.verb_embedding == "text":
                if verb_text_embeddings is None:
                    verb_text_embeddings = torch.zeros(cfg.num_verbs, cfg.verb_text_dim)
                self.register_buffer("verb_text", verb_text_embeddings.float())
                if cfg.verb_proj_deep:
                    self.verb_proj = nn.Sequential(
                        nn.Linear(cfg.verb_text_dim, 512), nn.ReLU(),
                        nn.Linear(512, 256), nn.ReLU(),
                        nn.Linear(256, cfg.verb_dim),
                    )
                else:
                    self.verb_proj = nn.Sequential(
                        nn.Linear(cfg.verb_text_dim, cfg.verb_dim), nn.ReLU(),
                        nn.Linear(cfg.verb_dim, cfg.verb_dim),
                    )
            else:
                self.verb_emb = nn.Embedding(cfg.num_verbs, cfg.verb_dim)

        self.use_global = cfg.global_dim > 0
        if self.use_global:
            self.global_encoder = nn.Sequential(
                nn.Linear(cfg.global_dim, cfg.cond_dim), nn.ReLU(),
                nn.Linear(cfg.cond_dim, cfg.cond_dim), nn.ReLU(),
            )
            self.film = nn.Linear(cfg.cond_dim, 2 * sum(cfg.hidden_dims))
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        if cfg.verb_conditioning == "cross_attn":
            self.verb_query = nn.Linear(cfg.verb_dim, cfg.hidden_dims[-1])
            self.attn_bias = nn.Parameter(torch.zeros(()))

    def _verb_vector(self, verb: int | torch.Tensor) -> torch.Tensor:
        if self.cfg.verb_embedding == "text":
            if torch.is_tensor(verb) and torch.is_floating_point(verb) and verb.shape[-1] == self.cfg.verb_text_dim:
                t = verb.to(self.verb_text.dtype).reshape(-1)
            else:
                idx = torch.as_tensor(verb, device=self.verb_text.device, dtype=torch.long)
                t = self.verb_text[idx].reshape(-1)
            return self.verb_proj(t).reshape(-1)
        idx = torch.as_tensor(verb, device=self.verb_emb.weight.device, dtype=torch.long)
        return self.verb_emb(idx).reshape(-1)

    def _per_vertex_input(
        self,
        verb_idx: int | torch.Tensor | None,
        slat_vertex: torch.Tensor | None,
        vlm_features: torch.Tensor | None,
        vertex_normals: torch.Tensor | None,
        vertex_positions: torch.Tensor | None,
        dino_vertex: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for t, dim, name in [
            (vlm_features, self.cfg.vlm_dim, "vlm_features"),
            (dino_vertex, self.cfg.dino_vertex_dim, "dino_vertex"),
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
        feat = torch.cat(parts, dim=-1)
        if self.cfg._verb_in_vertex:
            if verb_idx is None:
                raise ValueError("verb_idx required (verb_conditioning='concat')")
            v = self._verb_vector(verb_idx)
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
            parts.append(self._verb_vector(verb_idx))
        return torch.cat(parts, dim=-1).unsqueeze(0)

    def forward(
        self,
        verb_idx: int | torch.Tensor | None = None,
        slat_vertex: torch.Tensor | None = None,
        vlm_features: torch.Tensor | None = None,
        dino_cls: torch.Tensor | None = None,
        ss_dino_cls: torch.Tensor | None = None,
        vertex_normals: torch.Tensor | None = None,
        vertex_positions: torch.Tensor | None = None,
        dino_vertex: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self._per_vertex_input(verb_idx, slat_vertex, vlm_features, vertex_normals, vertex_positions, dino_vertex)
        if self.in_norm is not None:
            h = self.in_norm(h)

        film_params = None
        if self.use_global:
            g = self._global_input(verb_idx, dino_cls, ss_dino_cls)
            c = self.global_encoder(g)
            film_params = self.film(c)

        offset = 0
        for i, lin in enumerate(self.geom_layers):
            h = lin(h)
            if film_params is not None:
                H = self.cfg.hidden_dims[i]
                gamma = film_params[:, offset:offset + H]
                beta = film_params[:, offset + H:offset + 2 * H]
                offset += 2 * H
                h = h * (1.0 + gamma) + beta
            h = self.drop(self.act(h))
        if self.cfg.verb_conditioning == "cross_attn":
            q = self.verb_query(self._verb_vector(verb_idx))
            return (h @ q) / (h.shape[-1] ** 0.5) + self.attn_bias
        return self.out(h).squeeze(-1)


class AffordanceMLPHead(nn.Module):
    """Batch-dict wrapper around AffordanceMLP (iterates a collated batch list-of-tensors)."""

    def __init__(self, cfg: MLPHeadConfig) -> None:
        super().__init__()
        self.mlp = AffordanceMLP(cfg)

    def forward(self, batch: dict) -> list[torch.Tensor]:
        outputs = []
        cfg = self.mlp.cfg
        for i in range(len(batch["vertex_affordance"])):
            logits = self.mlp(
                verb_idx=batch["verb_idx"][i],
                vlm_features=batch["vertex_features"][i] if cfg.vlm_dim > 0 else None,
                slat_vertex=batch["slat_vertex_features"][i] if cfg.sam3d_dim > 0 else None,
                vertex_normals=batch["vertex_normals"][i] if cfg.normals_dim > 0 else None,
                vertex_positions=batch["vertex_positions"][i] if cfg.pos_dim > 0 else None,
                dino_cls=batch["dino_cls"][i] if cfg.dino_cls_dim > 0 else None,
                dino_vertex=batch["dino_vertex_features"][i] if cfg.dino_vertex_dim > 0 else None,
            )
            outputs.append(logits)
        return outputs


def mlp_head_config_from_model_cfg(model_cfg: dict[str, Any]) -> MLPHeadConfig:
    hd = model_cfg.get("hidden_dims", [256, 128])
    return MLPHeadConfig(
        vlm_dim=int(model_cfg.get("vlm_dim", 512)),
        sam3d_dim=int(model_cfg.get("sam3d_dim", 0)),
        normals_dim=int(model_cfg.get("normals_dim", 0)),
        pos_dim=int(model_cfg.get("pos_dim", 0)),
        dino_vertex_dim=int(model_cfg.get("dino_vertex_dim", 0)),
        dino_filename=str(model_cfg.get("dino_filename", "vertex_dino.pt")),
        verb_dim=int(model_cfg.get("verb_dim", 512)),
        num_verbs=int(model_cfg.get("num_verbs", 0)),
        dino_cls_dim=int(model_cfg.get("dino_cls_dim", 0)),
        ss_dino_cls_dim=int(model_cfg.get("ss_dino_cls_dim", 0)),
        cond_dim=int(model_cfg.get("cond_dim", 128)),
        hidden_dims=tuple(int(x) for x in hd),
        dropout=float(model_cfg.get("dropout", 0.1)),
        verb_conditioning=str(model_cfg.get("verb_conditioning", "concat")),
        input_layernorm=bool(model_cfg.get("input_layernorm", True)),
        verb_embedding=str(model_cfg.get("verb_embedding", "learned")),
        verb_text_dim=int(model_cfg.get("verb_text_dim", 512)),
        verb_proj_deep=bool(model_cfg.get("verb_proj_deep", False)),
    )
