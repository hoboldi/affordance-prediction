from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from datasets.data_root_dataset import DataRootDataset, ManifestRow, resolve_data_root, resolve_manifest_path
from datasets.data_root_dataset import load_manifest_rows


def _write_manifest(root: Path, lines: list[dict]) -> Path:
    mp = root / "manifest.jsonl"
    with mp.open("w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps(row) + "\n")
    return mp


def test_resolve_paths_relative_to_data_root(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "m").mkdir(parents=True)
    mesh = root / "m" / "a.obj"
    trimesh.creation.box(extents=[0.2, 0.2, 0.2]).export(mesh)

    _write_manifest(
        root,
        [{"sample_id": "a", "mesh_path": "m/a.obj", "split": "train"}],
    )

    rows = load_manifest_rows(root / "manifest.jsonl", data_root=root)
    assert len(rows) == 1
    assert rows[0].mesh_path == mesh


def test_split_filter(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "m").mkdir(parents=True)
    mesh = root / "m" / "a.obj"
    trimesh.creation.box().export(mesh)

    _write_manifest(
        root,
        [
            {"sample_id": "t", "mesh_path": "m/a.obj", "split": "train"},
            {"sample_id": "v", "mesh_path": "m/a.obj", "split": "val"},
        ],
    )

    train_rows = load_manifest_rows(root / "manifest.jsonl", data_root=root, split="train")
    assert [r.sample_id for r in train_rows] == ["t"]


def test_dataset_getitem_lazy_and_eager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    (root / "meshes").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    mesh_path = root / "meshes" / "x.obj"
    trimesh.creation.box().export(mesh_path)
    np.save(root / "labels" / "x.npy", np.linspace(0, 1, 8, dtype=np.float32))

    _write_manifest(
        root,
        [
            {
                "sample_id": "x",
                "verb": "open",
                "mesh_path": "meshes/x.obj",
                "vertex_affordance_path": "labels/x.npy",
                "split": "train",
            }
        ],
    )

    cfg = {
        "paths": {"data_root": str(root)},
        "dataset": {
            "manifest_filename": "manifest.jsonl",
            "load_mesh_eager": False,
            "load_vertex_labels_eager": False,
        },
    }

    ds_lazy = DataRootDataset(data_root=root, cfg=cfg)
    item = ds_lazy[0]
    assert item["mesh"] is None
    assert item["vertex_affordance"] is None
    assert item["mesh_path"] == mesh_path

    ds_eager = DataRootDataset(
        data_root=root,
        cfg=cfg,
        load_mesh_eager=True,
        load_vertex_labels_eager=True,
    )
    item2 = ds_eager[0]
    assert item2["mesh"] is not None
    assert item2["mesh"].num_vertices == 8
    assert item2["vertex_affordance"].shape == (8,)


def test_sam3d_latent_pt_dict(tmp_path: Path) -> None:
    """The latent is read from <sam3d_reconstruction_dir>/global_latent.pt and exposed as 'global_latent'."""
    root = tmp_path / "data"
    recon = root / "recon" / "z"
    recon.mkdir(parents=True)
    torch.save({"global_latent": torch.zeros(8)}, recon / "global_latent.pt")

    _write_manifest(
        root,
        [{"sample_id": "z", "sam3d_reconstruction_dir": "recon/z"}],
    )

    ds = DataRootDataset(data_root=root, manifest_path=root / "manifest.jsonl", cfg={}, load_vertex_labels_eager=False)
    z = ds[0]
    assert z["global_latent"].shape == (8,)


def test_env_affordance_data_root_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "other_data"
    override.mkdir()
    monkeypatch.setenv("AFFORDANCE_DATA_ROOT", str(override))
    cfg = {"paths": {"data_root": "data"}}
    assert resolve_data_root(cfg) == override.resolve()


def test_resolve_manifest_path_relative(tmp_path: Path) -> None:
    cfg = {"paths": {"data_root": str(tmp_path)}, "dataset": {"manifest_filename": "custom.jsonl"}}
    assert resolve_manifest_path(cfg, data_root=tmp_path) == tmp_path / "custom.jsonl"


def test_example_pack_in_repo() -> None:
    """Tracked example under examples/data_manifest must stay loadable."""
    root = Path(__file__).resolve().parents[1] / "examples" / "data_manifest"
    if not (root / "manifest.jsonl").exists():
        pytest.skip("example manifest not in tree")
    ds = DataRootDataset(
        data_root=root,
        manifest_path=root / "manifest.jsonl",
        cfg={},
        load_mesh_eager=True,
        load_vertex_labels_eager=True,
    )
    assert len(ds) == 1
    row = ds.row(0)
    assert isinstance(row, ManifestRow)
    item = ds[0]
    assert item["mesh"].num_vertices == 3
    assert item["vertex_affordance"].shape == (3,)


def test_training_manifest_with_fixtures() -> None:
    """Optional: full training row when fixture binaries exist (CI can run the script first)."""
    root = Path(__file__).resolve().parents[1] / "examples" / "data_manifest"
    feat = root / "features" / "tiny_vertex_semantic.pt"
    mt = root / "manifest_training.jsonl"
    if not (feat.is_file() and mt.is_file()):
        pytest.skip("run: python scripts/build_example_training_fixtures.py")
    ds = DataRootDataset(
        data_root=root,
        manifest_path=mt,
        cfg={},
        split="train",
        load_mesh_eager=False,
        load_vertex_labels_eager=True,
    )
    item = ds[0]
    assert item["vertex_features"].shape == (3, 512)
    assert item["sam3d_global_latent"].shape == (8,)
