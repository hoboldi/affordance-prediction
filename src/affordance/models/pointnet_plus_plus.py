import torch
import torch.nn as nn

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
        idx = torch.linspace(0, N - 1, self.n_out, device=xyz.device).long()
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


