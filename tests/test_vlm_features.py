from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")

from vlm.patch_extractor import extract_patch_features
from vlm.text_encoder import cosine_similarity, encode_texts
from vlm.vlm_wrapper import VLMWrapper

RENDER_CACHE = Path(__file__).resolve().parents[1] / "outputs" / "notebooks" / "02_rendering"


@pytest.fixture(scope="module")
def vlm() -> VLMWrapper:
    return VLMWrapper()


def test_text_embeddings_normalized(vlm: VLMWrapper):
    emb = encode_texts(vlm, ["grasp", "sit on"])
    assert emb.shape == (2, vlm.feature_dim)
    norms = emb.norm(dim=-1)
    assert np.allclose(norms.numpy(), 1.0, atol=1e-4)


def test_patch_features_shape(vlm: VLMWrapper):
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[32:96, 32:96] = 180
    feats = extract_patch_features(vlm, [rgb])[0]
    assert feats.num_patches == vlm.num_patches
    assert feats.patches.shape == (vlm.num_patches, vlm.feature_dim)
    assert feats.grid_h == feats.grid_w == vlm.patch_grid_size


@pytest.mark.skipif(not (RENDER_CACHE / "rgb_0.npy").exists(), reason="run 02_rendering notebook first")
def test_patch_features_on_cached_render(vlm: VLMWrapper):
    rgb = np.load(RENDER_CACHE / "rgb_0.npy")
    feats = extract_patch_features(vlm, [rgb])[0]
    assert feats.patches.shape[0] == vlm.num_patches


def test_verb_similarity_differs(vlm: VLMWrapper):
    texts = ["grasp", "pour from", "sky"]
    emb = encode_texts(vlm, texts)
    sim = cosine_similarity(emb, emb)
    assert sim[0, 1] > sim[0, 2]
