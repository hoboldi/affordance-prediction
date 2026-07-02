"""Learned encoder over SAM3D's structured latent (SLAT), for verb-conditioned affordance.

The pipeline currently consumes SLAT as a *flat* 8-d per-vertex channel (``slat_vertex_features``,
each vertex = its nearest voxel's raw latent). The ablation shows even that flat channel props up the
two appearance-hard verbs (display −0.24, grasp −0.10). :class:`SlatEncoder` *learns* from the SLAT
instead of reading it raw: EdgeConv message passing over a kNN graph of the ~8–9k SLAT voxels
(``slat_coords`` + ``slat_feats``) produces richer per-voxel features, which are then gathered to the
mesh vertices via a precomputed vertex→voxel map. Output: ``(V_mesh, out_dim)`` per-vertex 3D-structure
features to concatenate alongside DINO before the affordance head.

This reuses the EdgeConv formulation of :class:`models.gnn_head.AffordanceGNN`, but the SLAT graph is
tiny (thousands of voxels, not ~1M vertices) so it needs no chunking/checkpointing, and it is
verb-agnostic (a shape encoder; verb conditioning stays in the downstream head).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SlatEncoderConfig:
    in_dim: int = 8          # slat_feats channel dim (native SLAT = 8)
    hidden: int = 64         # EdgeConv width
    layers: int = 3          # EdgeConv layers
    out_dim: int = 64        # per-vertex feature dim handed to the affordance head
    knn_k: int = 8           # neighbours in the voxel graph
    dropout: float = 0.1
    use_coords: bool = True  # append normalized voxel xyz to the input latent


def build_slat_knn(slat_coords: torch.Tensor, k: int = 8) -> torch.Tensor:
    """kNN indices (Vv, k) over the SLAT voxels via scipy cKDTree; self-edge dropped. Long."""
    from scipy.spatial import cKDTree

    pos = slat_coords.detach().cpu().numpy()
    kk = min(k, len(pos) - 1)
    _, idx = cKDTree(pos).query(pos, k=kk + 1)
    idx = torch.as_tensor(idx[:, 1:], dtype=torch.long)          # drop self (column 0)
    if kk < k:                                                    # tiny cloud: pad by repeating last col
        idx = torch.cat([idx, idx[:, -1:].expand(-1, k - kk)], dim=1)
    return idx


def build_vertex_to_slat(vertex_positions: torch.Tensor, slat_coords: torch.Tensor) -> torch.Tensor:
    """(V_mesh,) index of the nearest SLAT voxel for each mesh vertex (cKDTree). Long.

    Mirrors how ``slat_vertex_features`` is assembled, so the encoder's gathered features are aligned
    with the mesh vertices the head predicts on.
    """
    from scipy.spatial import cKDTree

    _, idx = cKDTree(slat_coords.detach().cpu().numpy()).query(vertex_positions.detach().cpu().numpy(), k=1)
    return torch.as_tensor(idx, dtype=torch.long)


class SlatEncoder(nn.Module):
    """EdgeConv over the SLAT voxel graph → per-vertex 3D-structure features (verb-agnostic)."""

    def __init__(self, cfg: SlatEncoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SlatEncoderConfig()
        self.cfg = cfg
        in_d = cfg.in_dim + (3 if cfg.use_coords else 0)
        C = cfg.hidden
        self.in_norm = nn.LayerNorm(in_d)
        self.in_proj = nn.Linear(in_d, C)
        self.edge_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * C, C), nn.ReLU()) for _ in range(cfg.layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(C) for _ in range(cfg.layers))
        self.drop = nn.Dropout(cfg.dropout)
        self.out_proj = nn.Linear(C, cfg.out_dim)

    @staticmethod
    def _norm_coords(coords: torch.Tensor) -> torch.Tensor:
        """Zero-mean, scale by half the max extent → roughly [-1, 1], object-scale invariant."""
        c = coords - coords.mean(0, keepdim=True)
        h = (c.abs().amax() + 1e-6)
        return c / h

    def _edge_conv(self, h: torch.Tensor, knn: torch.Tensor, edge_mlp: nn.Module, ln: nn.Module) -> torch.Tensor:
        # h (Vv, C), knn (Vv, k) — SLAT graphs are small, so no chunking needed.
        k = knn.shape[1]
        center = h.unsqueeze(1).expand(-1, k, -1)                 # (Vv, k, C)
        hj = h[knn]                                               # (Vv, k, C)
        msg = edge_mlp(torch.cat([center, hj - center], dim=-1))  # (Vv, k, C)
        return ln(h + self.drop(msg.amax(dim=1)))                 # residual + LayerNorm

    def forward(
        self,
        slat_feats: torch.Tensor,       # (Vv, in_dim)
        slat_coords: torch.Tensor,      # (Vv, 3)
        slat_knn: torch.Tensor,         # (Vv, k) long  — precomputed voxel graph
        vertex_to_slat: torch.Tensor,   # (V_mesh,) long — nearest voxel per mesh vertex
    ) -> torch.Tensor:
        x = slat_feats.float()
        if self.cfg.use_coords:
            x = torch.cat([x, self._norm_coords(slat_coords.float())], dim=-1)
        h = self.in_proj(self.in_norm(x))                        # (Vv, C)
        knn = slat_knn.to(h.device, torch.long)
        for edge_mlp, ln in zip(self.edge_mlps, self.norms):
            h = self._edge_conv(h, knn, edge_mlp, ln)
        z = self.out_proj(h)                                     # (Vv, out_dim) encoded voxel features
        return z[vertex_to_slat.to(z.device, torch.long)]       # (V_mesh, out_dim) gathered to vertices
