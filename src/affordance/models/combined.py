import torch
import torch.nn as nn

from affordance.models.mlp_head import MLPHead, VERTEX_DIM
from affordance.models.pointnet_plus_plus import (
    FeaturePropagation,
    SetAbstraction,
    _interpolate,
)

_PN2_OUT_DIM = 128  # output channels from FP1


class _PN2Encoder(nn.Module):
    """PointNet++ encoder: full N vertices → per-vertex spatial features (N, _PN2_OUT_DIM)."""

    def __init__(self, n_sub: int, sa1_n: int, sa2_n: int, k: int):
        super().__init__()
        self.n_sub = n_sub
        self.sa1 = SetAbstraction(sa1_n, k, VERTEX_DIM, [64, 64, 128])
        self.sa2 = SetAbstraction(sa2_n, k, 128, [128, 128, 256])
        self.fp2 = FeaturePropagation(256 + 128, [256, 256], k=3)
        self.fp1 = FeaturePropagation(256 + VERTEX_DIM, [128, 128], k=3)

    def forward(self, xyz_list: list[torch.Tensor], feat_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        xyz_list:  list of (N_i, 3)    vertex positions per sample
        feat_list: list of (N_i, 143)  per-vertex features per sample
        returns:   list of (N_i, 128)  spatially-aggregated features
        """
        outputs = []
        for xyz, feat in zip(xyz_list, feat_list):
            N = xyz.shape[0]
            n_sub = min(self.n_sub, N)
            idx = torch.randperm(N, device=xyz.device)[:n_sub]
            sub_xyz, sub_feat = xyz[idx], feat[idx]

            sa1_xyz, sa1_feat = self.sa1(sub_xyz, sub_feat)
            sa2_xyz, sa2_feat = self.sa2(sa1_xyz, sa1_feat)
            fp2_feat = self.fp2(sa1_xyz, sa2_xyz, sa1_feat, sa2_feat)
            fp1_feat = self.fp1(sub_xyz, sa1_xyz, sub_feat, fp2_feat)

            outputs.append(_interpolate(sub_xyz, fp1_feat, xyz, k=3))  # (N, 128)
        return outputs


class PN2MLPHead(nn.Module):
    """
    MLPHead with PointNet++ spatial features as additional per-vertex input.

    Architecture:
        PN2Encoder  →  (N, 128) spatial features
        MLPHead     ←  standard per-vertex features + global + PN2 features
    """

    def __init__(
        self,
        n_sub: int = 4096,
        sa1_n: int = 512,
        sa2_n: int = 128,
        k: int = 16,
        verb_embedding_dim: int = 64,
        layer_dims: list[int] = [1024, 1024, 512, 256],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = _PN2Encoder(n_sub, sa1_n, sa2_n, k)
        self.head = MLPHead(
            layer_dims=layer_dims,
            verb_embedding_dim=verb_embedding_dim,
            dropout=dropout,
            extra_vertex_dim=_PN2_OUT_DIM,
        )

    def _build_per_vertex(self, vpos, vnorm, slat_vf, vsem_f, vsem_vis):
        return torch.cat([
            vpos, vnorm, slat_vf, vsem_f,
            vsem_vis.unsqueeze(-1).float(),
        ], dim=-1)  # (N, VERTEX_DIM)

    def forward(self, batch: dict) -> list[torch.Tensor]:
        xyz_list  = batch["vertex_positions"]
        feat_list = [
            self._build_per_vertex(vpos, vnorm, slat_vf, vsem_f, vsem_vis)
            for vpos, vnorm, slat_vf, vsem_f, vsem_vis in zip(
                batch["vertex_positions"], batch["vertex_normals"],
                batch["slat_vertex_features"], batch["vsem_features"],
                batch["vsem_visible"],
            )
        ]
        pn2_features = self.encoder(xyz_list, feat_list)       # list of (N_i, 128)
        return self.head(batch, extra_vertex_features=pn2_features)
