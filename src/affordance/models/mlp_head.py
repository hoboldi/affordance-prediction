import torch
import torch.nn as nn

from affordance.data.dataset import VERBS

# Per-vertex features
_VERTEX_POSITIONS_DIM  = 3
_VERTEX_NORMALS_DIM    = 3
_SLAT_VERTEX_FEAT_DIM  = 8
_VSEM_FEATURES_DIM     = 128
_VSEM_VISIBLE_DIM      = 1
VERTEX_DIM = (
    _VERTEX_POSITIONS_DIM
    + _VERTEX_NORMALS_DIM
    + _SLAT_VERTEX_FEAT_DIM
    + _VSEM_FEATURES_DIM
    + _VSEM_VISIBLE_DIM
)  # 143

# Global features (broadcast to each vertex)
_DINO_CLS_DIM           = 1024
_DINO_PATCHES_DIM       = 1024  # mean-pooled over patches
_SHAPE_LATENT_DIM       = 8     # mean-pooled over latents
_SLAT_FEATS_POOLED_DIM  = 8     # mean-pooled over sparse latents
_SLAT_COORDS_POOLED_DIM = 3     # mean-pooled over sparse latents
GLOBAL_DIM = (
    _DINO_CLS_DIM
    + _DINO_PATCHES_DIM
    + _SHAPE_LATENT_DIM
    + _SLAT_FEATS_POOLED_DIM
    + _SLAT_COORDS_POOLED_DIM
)  # 2067


class MLPHead(nn.Module):
    def __init__(
        self,
        layer_dims: list[int],
        verb_embedding_dim: int = 64,
        dropout: float = 0.0,
        extra_vertex_dim: int = 0,
    ):
        super().__init__()

        in_dim = VERTEX_DIM + extra_vertex_dim + GLOBAL_DIM + verb_embedding_dim
        self.verb_embedding = nn.Embedding(len(VERBS), verb_embedding_dim)

        dims = [in_dim] + layer_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            if dropout > 0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        batch: dict,
        extra_vertex_features: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        verb_emb            = self.verb_embedding(batch["verb_idx"])    # (B, verb_embedding_dim)
        dino_patches_pooled = batch["dino_patches"].mean(dim=1)         # (B, 1024)
        shape_latent_pooled = batch["shape_latent"].mean(dim=1)         # (B, 8)

        global_fixed = torch.cat([
            batch["dino_cls"],    # (B, 1024)
            dino_patches_pooled,  # (B, 1024)
            shape_latent_pooled,  # (B, 8)
            verb_emb,             # (B, verb_embedding_dim)
        ], dim=-1)

        outputs = []
        for i, (vpos, vnorm, slat_vf, vsem_f, vsem_vis, slat_c, slat_ft) in enumerate(zip(
            batch["vertex_positions"],
            batch["vertex_normals"],
            batch["slat_vertex_features"],
            batch["vsem_features"],
            batch["vsem_visible"],
            batch["slat_coords"],
            batch["slat_feats"],
        )):
            n = vpos.shape[0]
            slat_coords_pooled = slat_c.float().mean(dim=0, keepdim=True).expand(n, -1)  # (N, 3)
            slat_feats_pooled  = slat_ft.mean(dim=0, keepdim=True).expand(n, -1)         # (N, 8)

            parts = [
                vpos,                                         # (N, 3)
                vnorm,                                        # (N, 3)
                slat_vf,                                      # (N, 8)
                vsem_f,                                       # (N, 128)
                vsem_vis.unsqueeze(-1).float(),               # (N, 1)
            ]
            if extra_vertex_features is not None:
                parts.append(extra_vertex_features[i])        # (N, extra_vertex_dim)
            parts += [
                global_fixed[i].unsqueeze(0).expand(n, -1),  # (N, GLOBAL_DIM + verb_embedding_dim)
                slat_coords_pooled,                           # (N, 3)
                slat_feats_pooled,                            # (N, 8)
            ]
            x = torch.cat(parts, dim=-1)                      # (N, in_dim)

            outputs.append(self.mlp(x).squeeze(-1))  # (N,)

        return outputs
