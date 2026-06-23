from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from affordance.data.dataloader import _collate, get_dataloader
from affordance.data.dataset import VERBS, AffordanceDataset

DATA_ROOT = Path(__file__).parent.parent / "data"

# Skip all tests if the dataset is not yet downloaded
pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "manifest.pseudolabeled.vsem.jsonl").exists(),
    reason="Dataset not downloaded",
)


# --- Dataset construction ---

def test_single_has_one_sample():
    ds = AffordanceDataset.single(DATA_ROOT)
    assert len(ds) == 1


def test_nine_has_nine_samples():
    ds = AffordanceDataset.nine(DATA_ROOT)
    assert len(ds) == 9


def test_ninety_has_ninety_samples():
    ds = AffordanceDataset.ninety(DATA_ROOT)
    assert len(ds) == 90


def test_nine_has_equal_verb_distribution():
    ds = AffordanceDataset.nine(DATA_ROOT)
    counts = Counter(s["verb"] for s in ds.samples)
    assert set(counts.keys()) == set(VERBS)
    assert all(v == 1 for v in counts.values())


def test_ninety_has_equal_verb_distribution():
    ds = AffordanceDataset.ninety(DATA_ROOT)
    counts = Counter(s["verb"] for s in ds.samples)
    assert set(counts.keys()) == set(VERBS)
    assert all(v == 10 for v in counts.values())


def test_full_dataset_respects_split():
    train = AffordanceDataset(DATA_ROOT, split="train")
    val = AffordanceDataset(DATA_ROOT, split="val")
    train_ids = {s["sample_id"] for s in train.samples}
    val_ids = {s["sample_id"] for s in val.samples}
    assert len(train) > 0
    assert len(val) > 0
    assert train_ids.isdisjoint(val_ids)


# --- Sample content ---

@pytest.fixture(scope="module")
def one_sample():
    return AffordanceDataset.single(DATA_ROOT)[0]


def test_sample_has_required_keys(one_sample):
    required = {
        "sample_id", "verb", "verb_idx", "category",
        "vertex_positions", "vertex_normals",
        "slat_coords", "slat_feats", "slat_vertex_features",
        "dino_cls", "dino_patches", "shape_latent",
        "pseudolabels", "vsem_features", "vsem_visible",
    }
    assert required.issubset(one_sample.keys())


def test_fixed_size_tensors(one_sample):
    assert one_sample["dino_cls"].shape == (1024,)
    assert one_sample["dino_patches"].shape == (5495, 1024)
    assert one_sample["shape_latent"].shape == (4096, 8)


def test_vertex_tensors_are_consistent(one_sample):
    n = one_sample["vertex_positions"].shape[0]
    assert one_sample["vertex_normals"].shape == (n, 3)
    assert one_sample["slat_vertex_features"].shape == (n, 8)
    assert one_sample["pseudolabels"].shape == (n,)
    assert one_sample["vsem_features"].shape == (n, 128)
    assert one_sample["vsem_visible"].shape == (n,)


def test_slat_tensors_are_consistent(one_sample):
    m = one_sample["slat_coords"].shape[0]
    assert one_sample["slat_feats"].shape == (m, 8)


def test_verb_idx_is_valid(one_sample):
    assert isinstance(one_sample["verb_idx"], torch.Tensor)
    assert 0 <= one_sample["verb_idx"].item() < len(VERBS)
    assert VERBS[one_sample["verb_idx"].item()] == one_sample["verb"]


# --- Dataloader ---

@pytest.fixture(scope="module")
def one_batch():
    ds = AffordanceDataset.nine(DATA_ROOT)
    loader = get_dataloader(ds, batch_size=3, shuffle=False, num_workers=0)
    return next(iter(loader))


def test_batch_fixed_tensors_are_stacked(one_batch):
    assert one_batch["dino_cls"].shape == (3, 1024)
    assert one_batch["dino_patches"].shape == (3, 5495, 1024)
    assert one_batch["shape_latent"].shape == (3, 4096, 8)
    assert one_batch["verb_idx"].shape == (3,)


def test_batch_variable_tensors_are_lists(one_batch):
    for key in ("vertex_positions", "vertex_normals", "slat_coords",
                "slat_feats", "slat_vertex_features",
                "pseudolabels", "vsem_features", "vsem_visible"):
        assert isinstance(one_batch[key], list)
        assert len(one_batch[key]) == 3
        assert isinstance(one_batch[key][0], torch.Tensor)


def test_batch_metadata_are_lists(one_batch):
    assert isinstance(one_batch["sample_id"], list)
    assert isinstance(one_batch["verb"], list)
    assert isinstance(one_batch["category"], list)
    assert len(one_batch["sample_id"]) == 3
