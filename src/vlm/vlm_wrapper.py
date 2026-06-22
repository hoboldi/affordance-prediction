from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VLMConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
    max_batch_size: int = 8


def build_vlm_config(cfg: dict[str, Any]) -> VLMConfig:
    v = cfg.get("vlm", {})
    return VLMConfig(
        model_name=v.get("model_name") or "openai/clip-vit-base-patch32",
        device=v.get("device"),
        max_batch_size=int(v.get("max_batch_size", 8)),
    )


class VLMWrapper:
    """Frozen CLIP vision + text encoders (final ViT patch tokens)."""

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self.device = torch.device(
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )

        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
        # Safetensors avoids torch.load (transformers requires torch>=2.6 for pickle weights).
        self.model = CLIPModel.from_pretrained(self.config.model_name, use_safetensors=True)
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        vision_cfg = self.model.vision_model.config
        image_size = int(vision_cfg.image_size)
        patch_size = int(vision_cfg.patch_size)
        self.patch_grid_size = image_size // patch_size
        self.feature_dim = int(self.model.config.projection_dim)

    @property
    def num_patches(self) -> int:
        return self.patch_grid_size**2

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Frozen CLIP text embeddings, L2-normalized — (len(texts), projection_dim).

        Same CLIP space as the per-vertex image features, so a verb phrase can condition the head
        and generalize to unseen phrases (open-vocabulary verb conditioning).
        """
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            pooled = self.model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            ).pooler_output                                  # (N, hidden)
            feats = self.model.text_projection(pooled)        # (N, projection_dim)
        return (feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)).cpu()
