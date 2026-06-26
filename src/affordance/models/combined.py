import torch
import torch.nn as nn

from affordance.models.mlp_head import AffordanceMLP, MLPHeadConfig
from affordance.models.pointnet_plus_plus import FeaturePropagation, SetAbstraction

# Geometric features fed to the PN2 encoder: xyz (3) + normals (3) + slat vertex (8)
_PN2_GEO_DIM = 14
_PN2_OUT_DIM = 128


class _PN2Encoder(nn.Module):
    """PointNet++ encoder over geometric features → per-vertex spatial context (N, 128)."""

    def __init__(self, sa1_n: int, sa2_n: int, k: int):
        super().__init__()
        self.sa1 = SetAbstraction(sa1_n, k, _PN2_GEO_DIM, [64, 64, 128])
        self.sa2 = SetAbstraction(sa2_n, k, 128, [128, 128, 256])
        self.fp2 = FeaturePropagation(256 + 128, [256, 256], k=3)
        self.fp1 = FeaturePropagation(256 + _PN2_GEO_DIM, [128, 128], k=3)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        sa1_xyz, sa1_feat = self.sa1(xyz, feat)
        sa2_xyz, sa2_feat = self.sa2(sa1_xyz, sa1_feat)
        fp2_feat = self.fp2(sa1_xyz, sa2_xyz, sa1_feat, sa2_feat)
        return self.fp1(xyz, sa1_xyz, feat, fp2_feat)  # (N, 128)


class PN2MLPHead(nn.Module):
    """PointNet++ spatial encoder + AffordanceMLP head.

    The PN2 encoder aggregates geometric context (positions, normals, SLAT vertex features)
    into a 128-dim per-vertex feature, which is passed as ``dino_vertex`` to AffordanceMLP
    alongside the per-vertex VLM (CLIP) features and global conditioning.

    Forward signature matches AffordanceMLP so the same _forward() call works for both.
    """

    def __init__(self, cfg: MLPHeadConfig, sa1_n: int = 256, sa2_n: int = 64, k: int = 16):
        super().__init__()
        self.encoder = _PN2Encoder(sa1_n, sa2_n, k)
        self.head = AffordanceMLP(cfg)

    def forward(
        self,
        verb_idx=None,
        slat_vertex: torch.Tensor | None = None,
        vlm_features: torch.Tensor | None = None,
        dino_cls: torch.Tensor | None = None,
        ss_dino_cls: torch.Tensor | None = None,
        vertex_normals: torch.Tensor | None = None,
        vertex_positions: torch.Tensor | None = None,
        dino_vertex: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if vertex_positions is None:
            raise ValueError("vertex_positions required for PN2MLPHead")
        geo_parts = [vertex_positions]
        if vertex_normals is not None:
            geo_parts.append(vertex_normals)
        if slat_vertex is not None:
            geo_parts.append(slat_vertex)
        geo_feat = torch.cat(geo_parts, dim=-1)  # (N, _PN2_GEO_DIM)
        pn2_feat = self.encoder(vertex_positions, geo_feat)  # (N, 128)

        return self.head(
            verb_idx=verb_idx,
            vlm_features=vlm_features,
            dino_vertex=pn2_feat,
            dino_cls=dino_cls,
            ss_dino_cls=ss_dino_cls,
        )
