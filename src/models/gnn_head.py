"""Geometric GNN backbone for verb-conditioned per-vertex affordance prediction.

A drop-in alternative to :class:`models.mlp_head.AffordanceMLP`: instead of an FiLM-modulated
per-vertex MLP (which treats vertices independently), :class:`AffordanceGNN` runs EdgeConv-style
message passing over a kNN graph of the mesh vertices, so each vertex's prediction is informed by
its local geometric neighbourhood. Verb conditioning is the same OPEN-VOCAB "concat" mechanism as
the MLP head: a CLIP-text-derived verb vector is appended to every vertex before the output head.

Operates on ONE object at a time (no batch dim), V can be ~160k. Output: (V,) logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


@dataclass
class AffordanceGNNConfig:
    # Per-vertex (spatially-varying) inputs — same meaning as MLPHeadConfig.
    vlm_dim: int = 512
    dino_vertex_dim: int = 0     # per-vertex DINOv2 patch features (alongside CLIP vlm)
    geom_dim: int = 0            # per-vertex 3D geometry features (precompute_geom.py: height/concavity/curvature/normal_up/radial)
    sam3d_dim: int = 0           # per-vertex SAM3D SLAT features
    normals_dim: int = 0         # per-vertex surface normals
    pos_dim: int = 0             # normalized per-vertex xyz position
    dino_filename: str = "vertex_dino.pt"  # which per-recon-dir DINO file feeds dino_vertex (metadata)
    geom_filename: str = "vertex_geom.pt"  # which per-recon-dir geom file feeds vertex_geom (metadata)

    # Verb conditioning (open-vocab via CLIP text embedding + learnable projection), same as MLP head.
    verb_dim: int = 512
    num_verbs: int = 0
    verb_embedding: str = "text"   # "text" (open-vocab) | "learned" (closed set)
    verb_text_dim: int = 512
    verb_proj_deep: bool = False
    dropout: float = 0.1
    # Where the verb conditions the prediction:
    #   True  — concatenate the verb vector to the per-vertex features BEFORE the input projection, so
    #           the ENTIRE EdgeConv backbone is verb-conditioned (the message passing can select
    #           different regions per verb). Mirrors how the MLP injects the verb at the input.
    #   False — concatenate the verb only at the shallow output head (after a verb-agnostic backbone).
    #           This collapses verb-conditioning (every verb shares the backbone field; ep12 corr~0.94).
    verb_in_backbone: bool = True

    # Graph backbone.
    gnn_hidden: int = 128
    gnn_layers: int = 3
    knn_k: int = 8
    # Neighbour aggregation: "max" = EdgeConv max-pool (permutation-invariant, default) |
    #   "attn" = single-head graph attention (transformer-style) — each neighbour weighted by a
    #   learned query·key score, so the message passing can attend to the relevant neighbours per verb.
    agg: str = "max"
    # Augment EdgeConv messages (agg="max") with GEOMETRIC edge features: per neighbour, the relative
    # position (3), edge length (1) and surface-normal agreement (1) — scale-normalized by the object's
    # median nearest-neighbour distance. Gives the message passing explicit surface structure to reason
    # with (sharper region boundaries; targets the over-smoothed thin grasp band). Needs vertex_positions
    # + vertex_normals at forward. 5 extra input dims to each edge MLP.
    edge_geom: bool = False

    @property
    def edge_feat_dim(self) -> int:
        return 5 if (self.edge_geom and self.agg == "max") else 0
    # Memory bounding: EdgeConv messages are computed in row-chunks of this many vertices, and each
    # EdgeConv layer is gradient-checkpointed (activations recomputed in backward). This bounds peak
    # memory at O(gnn_chunk * k * C) regardless of total V (meshes here run up to ~1M+ vertices).
    gnn_chunk: int = 32768

    @property
    def per_vertex_dim(self) -> int:
        return self.vlm_dim + self.dino_vertex_dim + self.geom_dim + self.sam3d_dim + self.normals_dim + self.pos_dim

    @property
    def _verb_in_backbone(self) -> bool:
        return bool(self.verb_in_backbone) and self.verb_dim > 0

    @property
    def proj_in_dim(self) -> int:
        """Input width to in_norm/in_proj: per-vertex features (+ verb vector if verb_in_backbone)."""
        return self.per_vertex_dim + (self.verb_dim if self._verb_in_backbone else 0)

    @property
    def input_dim(self) -> int:
        """Total raw per-vertex feature dimensionality (for reporting)."""
        return self.per_vertex_dim


class AffordanceGNN(nn.Module):
    """Per-vertex affordance predictor over a kNN graph of mesh vertices.

    Pipeline (single object, V vertices):
        X = cat([vlm | dino | slat | normals | pos])  -> LayerNorm -> Linear(F -> C) = h
        for L EdgeConv layers:
            hj  = h[knn]                                       # (V, k, C) neighbour states
            msg = MLP( cat[h.expand, hj - h] )  (2C -> C, ReLU)
            agg = msg.amax(dim=1)                              # permutation-invariant
            h   = LayerNorm(h + dropout(agg))                  # residual
        v   = verb vector (verb_dim,)  (open-vocab concat)
        out = Linear(C + verb_dim -> C) -> ReLU -> Dropout -> Linear(C -> 1)  -> (V,)

    The kNN graph is supplied via ``knn_idx`` (precomputed) or built on-the-fly from
    ``vertex_positions`` with a scipy cKDTree (fallback / smoke-test path).
    """

    def __init__(
        self,
        cfg: AffordanceGNNConfig | None = None,
        verb_text_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or AffordanceGNNConfig()
        self.cfg = cfg

        if cfg.per_vertex_dim == 0:
            raise ValueError("At least one per-vertex input (vlm_dim, dino_vertex_dim, sam3d_dim, normals_dim, pos_dim) must be > 0")

        C = cfg.gnn_hidden

        # Balance scales across the concatenated per-vertex input (CLIP unit-norm vs DINO vs geom vs
        # slat/normals/pos). When verb_in_backbone, the verb vector is concatenated here too, so the
        # whole EdgeConv backbone is verb-conditioned (input width = proj_in_dim).
        self.in_norm = nn.LayerNorm(cfg.proj_in_dim)
        self.in_proj = nn.Linear(cfg.proj_in_dim, C)

        # Neighbour-aggregation parameters (one set per layer) + residual LayerNorms.
        #   "max"  — EdgeConv message MLP: Linear(2C->C), ReLU, then amax over neighbours.
        #   "attn" — single-head graph attention: per-layer query/key/value projections; neighbours
        #            weighted by softmax(q·k/sqrt(C)) then value-summed (transformer-style aggregation).
        if cfg.agg == "attn":
            self.attn_q = nn.ModuleList(nn.Linear(C, C) for _ in range(cfg.gnn_layers))
            self.attn_k = nn.ModuleList(nn.Linear(C, C) for _ in range(cfg.gnn_layers))
            self.attn_v = nn.ModuleList(nn.Linear(C, C) for _ in range(cfg.gnn_layers))
        else:
            self.edge_mlps = nn.ModuleList(
                nn.Sequential(nn.Linear(2 * C + cfg.edge_feat_dim, C), nn.ReLU()) for _ in range(cfg.gnn_layers)
            )
        self.layer_norms = nn.ModuleList(nn.LayerNorm(C) for _ in range(cfg.gnn_layers))
        self.drop = nn.Dropout(cfg.dropout)

        # ── Verb conditioning vector source (copied verbatim from AffordanceMLP) ──
        if cfg.verb_dim > 0:
            if cfg.num_verbs <= 0:
                raise ValueError("num_verbs must be > 0 when verb_dim > 0")
            if cfg.verb_embedding == "text":
                # Frozen CLIP text embedding per training verb (zeros here at load-time; filled by state_dict),
                # mapped to verb_dim by a small learnable projection. New verbs: pass a text embedding to forward.
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

        # ── Output head ──
        # If the verb is already injected into the backbone (verb_in_backbone), the head is verb-free
        # (Linear C->C->1) — the conditioning has already shaped the per-vertex features. Otherwise the
        # verb vector is concatenated only here (legacy shallow conditioning; collapses verb-distinctness).
        head_in = C if cfg._verb_in_backbone else (C + cfg.verb_dim)
        self.head = nn.Sequential(
            nn.Linear(head_in, C), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(C, 1),
        )
        nn.init.constant_(self.head[-1].bias, 0.0)  # balanced sampling → start at logit 0

    def _verb_vector(self, verb: int | torch.Tensor) -> torch.Tensor:
        """(verb_dim,) conditioning vector for a verb — by id, or directly from a CLIP text embedding.

        In ``verb_embedding="text"`` mode, pass a float tensor of shape (verb_text_dim,) to condition on an
        arbitrary/unseen verb phrase (open-vocabulary); pass an int id to use a training verb's stored embedding.
        """
        if self.cfg.verb_embedding == "text":
            if torch.is_tensor(verb) and torch.is_floating_point(verb) and verb.shape[-1] == self.cfg.verb_text_dim:
                t = verb.to(self.verb_text.dtype).reshape(-1)            # a CLIP text embedding (e.g. an unseen verb)
            else:
                idx = torch.as_tensor(verb, device=self.verb_text.device, dtype=torch.long)
                t = self.verb_text[idx].reshape(-1)
            return self.verb_proj(t).reshape(-1)
        idx = torch.as_tensor(verb, device=self.verb_emb.weight.device, dtype=torch.long)
        return self.verb_emb(idx).reshape(-1)

    def _per_vertex_input(
        self,
        vlm_features: torch.Tensor | None,
        dino_vertex: torch.Tensor | None,
        slat_vertex: torch.Tensor | None,
        vertex_normals: torch.Tensor | None,
        vertex_positions: torch.Tensor | None,
        vertex_geom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for t, dim, name in [
            (vlm_features, self.cfg.vlm_dim, "vlm_features"),
            (dino_vertex, self.cfg.dino_vertex_dim, "dino_vertex"),
            (vertex_geom, self.cfg.geom_dim, "vertex_geom"),
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
        return torch.cat(parts, dim=-1)  # (V, per_vertex_dim) — verb NOT appended here

    def _mp_layer(
        self,
        h: torch.Tensor,        # (V, C)
        knn_idx: torch.Tensor,  # (V, k) long
        li: int,                # layer index (picks per-layer modules)
    ) -> torch.Tensor:
        """One message-passing layer, computed in row-chunks over vertices to bound peak memory.

        agg="max" (EdgeConv, identical to the un-chunked form):
            hj  = h[knn]                                  # (chunk, k, C)
            msg = edge_mlp(cat[center, hj - center])      # (chunk, k, C)
            agg = msg.amax(dim=1)                         # (chunk, C)
        agg="attn" (single-head graph attention over the k neighbours):
            q=Wq(center); kk=Wk(hj); vv=Wv(hj)
            a   = softmax(sum(q*kk,-1)/sqrt(C), dim=1)    # (chunk, k)
            agg = sum(a*vv, dim=1)                         # (chunk, C)
        Then residual: out = ln(h + dropout(agg)). Only the (chunk, k, ·) intermediate is
        materialized at a time, so peak is O(chunk*k*C). Designed to run inside gradient checkpointing.
        """
        V, C = h.shape
        k = knn_idx.shape[1]
        ln = self.layer_norms[li]
        chunk = self.cfg.gnn_chunk if self.cfg.gnn_chunk and self.cfg.gnn_chunk > 0 else V
        agg = h.new_empty((V, C))
        attn = self.cfg.agg == "attn"
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            hc = h[s:e]                                              # (m, C)
            hj = h[knn_idx[s:e]]                                     # (m, k, C)
            center = hc.unsqueeze(1).expand(-1, k, -1)               # (m, k, C)
            if attn:
                q = self.attn_q[li](hc).unsqueeze(1)                 # (m, 1, C)
                kk = self.attn_k[li](hj)                             # (m, k, C)
                vv = self.attn_v[li](hj)                             # (m, k, C)
                a = torch.softmax((q * kk).sum(-1) / (C ** 0.5), dim=1)  # (m, k)
                agg[s:e] = (a.unsqueeze(-1) * vv).sum(1)             # (m, C)
            else:
                parts = [center, hj - center]
                if self.cfg.edge_feat_dim:                           # geometric edge features
                    pc = self._epos[s:e]; pj = self._epos[knn_idx[s:e]]         # (m,C? no: m,3)/(m,k,3)
                    rel = (pj - pc.unsqueeze(1)) / self._escale                 # (m,k,3) scale-normalized
                    dist = rel.norm(dim=-1, keepdim=True)                       # (m,k,1)
                    nc = self._enrm[s:e]; nj = self._enrm[knn_idx[s:e]]
                    ndot = (nc.unsqueeze(1) * nj).sum(-1, keepdim=True)         # (m,k,1) normal agreement
                    parts += [rel, dist, ndot]
                msg = self.edge_mlps[li](torch.cat(parts, dim=-1))  # (m, k, 2C[+5])
                agg[s:e] = msg.amax(dim=1)                           # (m, C)
        return ln(h + self.drop(agg))                                # residual + LayerNorm

    @staticmethod
    def _build_knn(vertex_positions: torch.Tensor, k: int) -> torch.Tensor:
        """kNN indices (V, k) from positions via scipy cKDTree, self-edge dropped. Long, same device."""
        from scipy.spatial import cKDTree

        pos = vertex_positions.detach().cpu().numpy()
        tree = cKDTree(pos)
        # query k+1 (the first hit is the point itself); drop column 0.
        _, idx = tree.query(pos, k=k + 1)
        idx = torch.as_tensor(idx[:, 1:], dtype=torch.long)  # (V, k)
        return idx.to(vertex_positions.device)

    def forward(
        self,
        verb_idx: int | torch.Tensor | None = None,
        vlm_features: torch.Tensor | None = None,
        dino_vertex: torch.Tensor | None = None,
        slat_vertex: torch.Tensor | None = None,
        vertex_normals: torch.Tensor | None = None,
        vertex_positions: torch.Tensor | None = None,
        vertex_geom: torch.Tensor | None = None,
        knn_idx: torch.Tensor | None = None,
        dino_cls: torch.Tensor | None = None,      # ignored — call-site compat
        ss_dino_cls: torch.Tensor | None = None,   # ignored — call-site compat
    ) -> torch.Tensor:
        # (a) assemble per-vertex features
        X = self._per_vertex_input(vlm_features, dino_vertex, slat_vertex, vertex_normals, vertex_positions, vertex_geom)

        # Verb vector (computed once; reused at input and/or head depending on verb_in_backbone).
        v = self._verb_vector(verb_idx) if self.cfg.verb_dim > 0 else None  # (verb_dim,)

        # (b) input projection. verb_in_backbone: concat the verb to EVERY vertex before in_proj so the
        # entire EdgeConv backbone is verb-conditioned (fixes the shallow-head verb collapse).
        if self.cfg._verb_in_backbone:
            if v is None:
                raise ValueError("verb required (verb_in_backbone=True, verb_dim>0)")
            X = torch.cat([X, v.unsqueeze(0).expand(X.shape[0], -1)], dim=-1)
        X = self.in_norm(X)
        h = self.in_proj(X)  # (V, C)

        # (c) graph
        if knn_idx is None:
            if vertex_positions is None:
                raise ValueError("knn_idx is None and vertex_positions is None — cannot build the kNN graph")
            knn_idx = self._build_knn(vertex_positions, self.cfg.knn_k)
        else:
            knn_idx = knn_idx.to(device=h.device, dtype=torch.long)

        # geometric edge features (scale-normalized): stash positions/normals + median edge length for _mp_layer
        if self.cfg.edge_feat_dim:
            if vertex_positions is None or vertex_normals is None:
                raise ValueError("edge_geom=True requires vertex_positions and vertex_normals at forward")
            self._epos = vertex_positions.to(device=h.device, dtype=torch.float32)
            nrm = vertex_normals.to(device=h.device, dtype=torch.float32)
            self._enrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-8)
            self._escale = (self._epos[knn_idx[:, 0]] - self._epos).norm(dim=-1).median().clamp(min=1e-6)

        # (d) EdgeConv-style message passing — each layer is computed in row-chunks (bounds the
        # (chunk,k,2C) intermediate) and gradient-checkpointed (the (chunk,k,2C) activations are
        # recomputed in backward rather than stored), so peak memory is bounded regardless of V.
        use_ckpt = torch.is_grad_enabled() and h.requires_grad
        for li in range(self.cfg.gnn_layers):
            if use_ckpt:
                h = cp.checkpoint(self._mp_layer, h, knn_idx, li, use_reentrant=False)
            else:
                h = self._mp_layer(h, knn_idx, li)

        # (e) output head. When the verb is already in the backbone, the head is verb-free; otherwise
        # concat the verb here (legacy shallow conditioning).
        if not self.cfg._verb_in_backbone and v is not None:
            h = torch.cat([h, v.unsqueeze(0).expand(h.shape[0], -1)], dim=-1)
        return self.head(h).squeeze(-1)                                 # (V,)


def gnn_head_config_from_model_cfg(model_cfg: dict[str, Any]) -> AffordanceGNNConfig:
    """Build :class:`AffordanceGNNConfig` from a ``configs/*.yaml`` ``model:`` block (+ gnn_* keys)."""
    return AffordanceGNNConfig(
        vlm_dim=int(model_cfg.get("vlm_dim", 512)),
        dino_vertex_dim=int(model_cfg.get("dino_vertex_dim", 0)),
        geom_dim=int(model_cfg.get("geom_dim", 0)),
        sam3d_dim=int(model_cfg.get("sam3d_dim", 0)),
        normals_dim=int(model_cfg.get("normals_dim", 0)),
        pos_dim=int(model_cfg.get("pos_dim", 0)),
        dino_filename=str(model_cfg.get("dino_filename", "vertex_dino.pt")),
        geom_filename=str(model_cfg.get("geom_filename", "vertex_geom.pt")),
        verb_dim=int(model_cfg.get("verb_dim", 512)),
        num_verbs=int(model_cfg.get("num_verbs", 0)),
        verb_embedding=str(model_cfg.get("verb_embedding", "text")),
        verb_text_dim=int(model_cfg.get("verb_text_dim", 512)),
        verb_proj_deep=bool(model_cfg.get("verb_proj_deep", False)),
        verb_in_backbone=bool(model_cfg.get("verb_in_backbone", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        gnn_hidden=int(model_cfg.get("gnn_hidden", 128)),
        gnn_layers=int(model_cfg.get("gnn_layers", 3)),
        knn_k=int(model_cfg.get("knn_k", 8)),
        agg=str(model_cfg.get("agg", "max")),
        edge_geom=bool(model_cfg.get("edge_geom", False)),
        gnn_chunk=int(model_cfg.get("gnn_chunk", 32768)),
    )


def build_affordance_gnn(
    cfg: dict[str, Any] | None = None,
    *,
    verb_text_embeddings: torch.Tensor | None = None,
) -> AffordanceGNN:
    """Instantiate :class:`AffordanceGNN` from merged project config (``model.*`` keys + gnn_* keys).

    Mirrors :func:`models.mlp_head.build_affordance_mlp`.
    """
    if cfg is None:
        from utils.config import load_config

        cfg = load_config()
    mcfg = gnn_head_config_from_model_cfg(dict(cfg.get("model", {})))
    return AffordanceGNN(mcfg, verb_text_embeddings=verb_text_embeddings)
