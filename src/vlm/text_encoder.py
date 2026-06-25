from __future__ import annotations

import torch

from vlm.vlm_wrapper import VLMWrapper


def encode_texts(wrapper: VLMWrapper, texts: list[str]) -> torch.Tensor:
    """
    L2-normalized CLIP text embeddings.

    Returns:
        Tensor of shape (len(texts), feature_dim) on CPU.
    """
    if not texts:
        return torch.empty(0, wrapper.feature_dim)

    with torch.inference_mode():
        inputs = wrapper.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(wrapper.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(wrapper.device)

        text_out = wrapper.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = text_out.pooler_output
        if pooled is None:
            pooled = text_out.last_hidden_state[:, -1, :]
        embeddings = wrapper.model.text_projection(pooled)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return embeddings.cpu()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between rows of ``a`` and ``b``."""
    a_norm = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b_norm = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return a_norm @ b_norm.T
