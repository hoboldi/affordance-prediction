import torch
import torch.nn as nn

from affordance.data.dataset import VERBS
from affordance.models.mlp_head import GLOBAL_DIM, VERTEX_DIM

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _knn(src_xyz: torch.Tensor, query_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Returns (M, k) indices into src_xyz for each query point."""
    dists = torch.cdist(query_xyz, src_xyz)           # (M, N)
    return dists.topk(k, dim=-1, largest=False).indices  # (M, k)


def _interpolate(src_xyz: torch.Tensor, src_feat: torch.Tensor,
                 tgt_xyz: torch.Tensor, k: int = 3,
                 chunk_size: int = 4096) -> torch.Tensor:
    """
    Inverse-distance-weighted interpolation from src → tgt.
    Chunked to avoid OOM when tgt is large (e.g. 160k vertices).
    """
    M = tgt_xyz.shape[0]
    C = src_feat.shape[1]
    out = torch.empty(M, C, device=src_feat.device, dtype=src_feat.dtype)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        dists = torch.cdist(tgt_xyz[start:end], src_xyz)       # (chunk, N)
        knn_d, knn_i = dists.topk(k, dim=-1, largest=False)    # (chunk, k)
        w = 1.0 / (knn_d + 1e-8)
        w = w / w.sum(-1, keepdim=True)                         # (chunk, k)
        out[start:end] = (w.unsqueeze(-1) * src_feat[knn_i]).sum(1)
    return out


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp_block(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]),
                   nn.BatchNorm1d(dims[i + 1]),
                   nn.ReLU()]
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    """Subsample → group k-NN → shared MLP → max pool."""

    def __init__(self, n_out: int, k: int, in_channels: int, mlp_dims: list[int]):
        super().__init__()
        self.n_out = n_out
        self.k = k
        self.mlp = _mlp_block([in_channels + 3] + mlp_dims)  # +3 for relative xyz
        self.out_channels = mlp_dims[-1]

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        """
        xyz:  (N, 3)
        feat: (N, C)
        returns: new_xyz (n_out, 3), new_feat (n_out, out_channels)
        """
        N = xyz.shape[0]
        idx = torch.randperm(N, device=xyz.device)[:self.n_out]
        new_xyz = xyz[idx]                                     # (n_out, 3)

        knn_idx = _knn(xyz, new_xyz, self.k)                  # (n_out, k)
        rel_xyz = xyz[knn_idx] - new_xyz.unsqueeze(1)         # (n_out, k, 3)
        grouped = torch.cat([rel_xyz, feat[knn_idx]], dim=-1) # (n_out, k, C+3)

        M, K, D = grouped.shape
        out = self.mlp(grouped.view(M * K, D)).view(M, K, -1) # (n_out, k, out)
        return new_xyz, out.max(dim=1).values                  # (n_out, out_channels)


class FeaturePropagation(nn.Module):
    """Interpolate sparse → dense, concat skip, shared MLP."""

    def __init__(self, in_channels: int, mlp_dims: list[int], k: int = 3):
        super().__init__()
        self.k = k
        self.mlp = _mlp_block([in_channels] + mlp_dims)
        self.out_channels = mlp_dims[-1]

    def forward(self, dense_xyz, sparse_xyz, dense_skip, sparse_feat):
        """
        dense_xyz:   (N, 3)   target resolution
        sparse_xyz:  (M, 3)   source resolution
        dense_skip:  (N, C1)  skip features from the SA at this level (or None)
        sparse_feat: (M, C2)  features to upsample
        returns: (N, out_channels)
        """
        interp = _interpolate(sparse_xyz, sparse_feat, dense_xyz, k=self.k)
        combined = torch.cat([dense_skip, interp], dim=-1) if dense_skip is not None else interp
        return self.mlp(combined)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PointNetPlusPlusHead(nn.Module):
    """
    PointNet++ encoder with hierarchical SA + FP, then per-vertex output MLP
    conditioned on global features (DINO, shape latent, verb).

    Processing pipeline:
      full N vertices
        → random subsample to n_sub
          → SA1 (n_sub → sa1_n)
            → SA2 (sa1_n → sa2_n)
          ← FP2 (sa2 → sa1 resolution)
        ← FP1 (sa1 → n_sub resolution)
      ← FP0 chunked interpolation (n_sub → full N)
      → concat global features → output MLP → (N,)
    """

    def __init__(
        self,
        n_sub: int = 4096,     # random subsample before SA
        sa1_n: int = 512,
        sa2_n: int = 128,
        k: int = 16,
        verb_embedding_dim: int = 64,
    ):
        super().__init__()
        self.n_sub = n_sub
        self.verb_embedding = nn.Embedding(len(VERBS), verb_embedding_dim)

        # per-vertex feature dim (same as mlp_head)
        pv = VERTEX_DIM  # 143

        self.sa1 = SetAbstraction(sa1_n, k, pv, [64, 64, 128])
        self.sa2 = SetAbstraction(sa2_n, k, 128, [128, 128, 256])

        self.fp2 = FeaturePropagation(256 + 128, [256, 256], k=3)
        self.fp1 = FeaturePropagation(256 + pv,  [128, 128], k=3)
        # FP0: chunked interpolation only (no learned MLP) — sub → full N

        global_dim = GLOBAL_DIM + verb_embedding_dim  # 2067 + 64 = 2131

        self.output_mlp = nn.Sequential(
            nn.Linear(128 + global_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _build_per_vertex(self, vpos, vnorm, slat_vf, vsem_f, vsem_vis):
        return torch.cat([
            vpos, vnorm, slat_vf, vsem_f,
            vsem_vis.unsqueeze(-1).float(),
        ], dim=-1)  # (N, 143)

    def forward(self, batch: dict) -> list[torch.Tensor]:
        verb_emb            = self.verb_embedding(batch["verb_idx"])
        dino_patches_pooled = batch["dino_patches"].mean(dim=1)
        shape_latent_pooled = batch["shape_latent"].mean(dim=1)

        outputs = []
        for i, (vpos, vnorm, slat_vf, vsem_f, vsem_vis, slat_c, slat_ft) in enumerate(zip(
            batch["vertex_positions"], batch["vertex_normals"],
            batch["slat_vertex_features"], batch["vsem_features"],
            batch["vsem_visible"], batch["slat_coords"], batch["slat_feats"],
        )):
            N = vpos.shape[0]

            # Global feature vector (same for every vertex in this sample)
            slat_coords_pooled = slat_c.float().mean(0)
            slat_feats_pooled  = slat_ft.mean(0)
            global_feat = torch.cat([
                batch["dino_cls"][i], dino_patches_pooled[i],
                shape_latent_pooled[i], slat_coords_pooled,
                slat_feats_pooled, verb_emb[i],
            ])  # (global_dim,)

            # Per-vertex features
            pv_feat = self._build_per_vertex(vpos, vnorm, slat_vf, vsem_f, vsem_vis)  # (N, 143)

            # --- Subsample ---
            n_sub = min(self.n_sub, N)
            sub_idx = torch.randperm(N, device=vpos.device)[:n_sub]
            sub_xyz  = vpos[sub_idx]    # (n_sub, 3)
            sub_feat = pv_feat[sub_idx] # (n_sub, 143)

            # --- Encoder ---
            sa1_xyz, sa1_feat = self.sa1(sub_xyz, sub_feat)   # (sa1_n, 128)
            sa2_xyz, sa2_feat = self.sa2(sa1_xyz, sa1_feat)   # (sa2_n, 256)

            # --- Decoder ---
            fp2_feat = self.fp2(sa1_xyz, sa2_xyz, sa1_feat, sa2_feat)  # (sa1_n, 256)
            fp1_feat = self.fp1(sub_xyz, sa1_xyz, sub_feat, fp2_feat)  # (n_sub, 128)

            # --- Upsample to full N (chunked) ---
            full_feat = _interpolate(sub_xyz, fp1_feat, vpos, k=3)  # (N, 128)

            # --- Output MLP ---
            g_exp = global_feat.unsqueeze(0).expand(N, -1)           # (N, global_dim)
            x = torch.cat([full_feat, g_exp], dim=-1)                # (N, 128+global_dim)
            outputs.append(self.output_mlp(x).squeeze(-1))           # (N,)

        return outputs
