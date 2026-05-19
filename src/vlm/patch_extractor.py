from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from vlm.vlm_wrapper import VLMWrapper


@dataclass
class PatchFeatures:
    """Patch tokens from the final ViT layer (CLS excluded)."""

    patches: torch.Tensor  # (N_patches, D) on CPU
    grid_h: int
    grid_w: int
    feature_dim: int

    @property
    def num_patches(self) -> int:
        return int(self.patches.shape[0])


def extract_patch_features(
    wrapper: VLMWrapper,
    images: list[np.ndarray],
) -> list[PatchFeatures]:
    """
    Extract final-layer ViT patch embeddings for each RGB image (uint8 HWC).
    """
    if not images:
        return []

    pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    all_patches: list[PatchFeatures] = []
    batch_size = wrapper.config.max_batch_size

    with torch.inference_mode():
        for start in range(0, len(pil_images), batch_size):
            batch = pil_images[start : start + batch_size]
            inputs = wrapper.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(wrapper.device)

            vision_out = wrapper.model.vision_model(pixel_values=pixel_values)
            hidden = vision_out.last_hidden_state  # (B, 1 + N, H)
            patch_tokens = hidden[:, 1:, :]  # drop CLS
            projected = wrapper.model.visual_projection(patch_tokens)  # (B, N, D)

            for i in range(projected.shape[0]):
                patches = projected[i].cpu()
                all_patches.append(
                    PatchFeatures(
                        patches=patches,
                        grid_h=wrapper.patch_grid_size,
                        grid_w=wrapper.patch_grid_size,
                        feature_dim=patches.shape[-1],
                    )
                )

    return all_patches


def encode_image_features(wrapper: VLMWrapper, images: list[np.ndarray]) -> torch.Tensor:
    """
    Global CLIP image embeddings (L2-normalized), shape ``(len(images), D)``.

    Use for image–text sanity checks; patch tokens are used for 2D→3D projection.
    """
    if not images:
        return torch.empty(0, wrapper.feature_dim)

    pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    batch_size = wrapper.config.max_batch_size
    chunks: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(pil_images), batch_size):
            batch = pil_images[start : start + batch_size]
            inputs = wrapper.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(wrapper.device)

            vision_out = wrapper.model.vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output
            if pooled is None:
                pooled = vision_out.last_hidden_state[:, 0, :]
            features = wrapper.model.visual_projection(pooled)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            chunks.append(features.cpu())

    return torch.cat(chunks, dim=0)
