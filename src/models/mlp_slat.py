"""AffordanceMLP + learned SLAT encoder.

Replaces the flat 8-d SLAT channel (``slat_vertex_features``, nearest-voxel projection) with a learned
:class:`models.slat_encoder.SlatEncoder` over SAM3D's structured latent. The encoder's per-vertex output
(``slat_cfg.out_dim``) is fed as the MLP's ``slat_vertex`` input, so ``mlp_cfg.sam3d_dim`` must equal
``slat_cfg.out_dim``. Everything else in :class:`models.mlp_head.AffordanceMLP` is unchanged (DINO,
CLIP, normals, cls tokens, verb conditioning, head).

The added encoder means the input projection changes shape, so this cannot be head-only fine-tuned from
the flat-SLAT base — :func:`load_base_into_slat_model` warm-starts every matching param (hidden layers,
verb projection, head, verb-text buffer) and leaves the reshaped input layers + the encoder fresh.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.mlp_head import AffordanceMLP, MLPHeadConfig
from models.slat_encoder import SlatEncoder, SlatEncoderConfig


class AffordanceMLPSlat(nn.Module):
    def __init__(
        self,
        mlp_cfg: MLPHeadConfig,
        slat_cfg: SlatEncoderConfig,
        verb_text_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if int(mlp_cfg.sam3d_dim) != int(slat_cfg.out_dim):
            raise ValueError(
                f"mlp_cfg.sam3d_dim ({mlp_cfg.sam3d_dim}) must equal slat_cfg.out_dim ({slat_cfg.out_dim}) "
                "— the encoder output is fed as the MLP's slat_vertex channel"
            )
        self.slat_cfg = slat_cfg
        self.slat_encoder = SlatEncoder(slat_cfg)
        self.mlp = AffordanceMLP(mlp_cfg, verb_text_embeddings=verb_text_embeddings)

    def forward(
        self,
        verb_idx,
        *,
        slat_feats: torch.Tensor,
        slat_coords: torch.Tensor,
        slat_knn: torch.Tensor,
        vertex_to_slat: torch.Tensor,
        vertex_weights: torch.Tensor | None = None,
        vlm_features: torch.Tensor | None = None,
        dino_cls: torch.Tensor | None = None,
        ss_dino_cls: torch.Tensor | None = None,
        vertex_normals: torch.Tensor | None = None,
        vertex_positions: torch.Tensor | None = None,
        dino_vertex: torch.Tensor | None = None,
        vertex_geom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slat_enc = self.slat_encoder(slat_feats, slat_coords, slat_knn, vertex_to_slat, vertex_weights)  # (V, out_dim)
        return self.mlp(
            verb_idx,
            slat_vertex=slat_enc,
            vlm_features=vlm_features,
            dino_cls=dino_cls,
            ss_dino_cls=ss_dino_cls,
            vertex_normals=vertex_normals,
            vertex_positions=vertex_positions,
            dino_vertex=dino_vertex,
            vertex_geom=vertex_geom,
        )


def load_base_into_slat_model(model: AffordanceMLPSlat, base_ckpt_path: str) -> tuple[int, int]:
    """Warm-start ``model.mlp`` from a flat-SLAT base checkpoint: copy every param whose shape matches
    (hidden layers, verb_proj, head, verb_text), skip the reshaped input norm/proj. The SlatEncoder and
    the reshaped input layers stay freshly initialized. Returns (num_loaded, num_skipped)."""
    base = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)["model"]
    tgt = model.mlp.state_dict()
    loaded, skipped = 0, 0
    for k, v in base.items():
        if k in tgt and tuple(tgt[k].shape) == tuple(v.shape):
            tgt[k] = v
            loaded += 1
        else:
            skipped += 1
    model.mlp.load_state_dict(tgt)
    return loaded, skipped
