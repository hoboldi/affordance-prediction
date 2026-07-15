from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, MLPHeadConfig
from training.vertex_affordance_train import eval_vertex_bce, training_epoch_vertex_bce


def test_vertex_semantics_via_dataset(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "f").mkdir(parents=True)
    p = root / "f" / "sem.pt"
    torch.save(
        {"features": torch.randn(5, 16), "visible_in_any_view": torch.tensor([True, False, True, True, False])},
        p,
    )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        '{"sample_id": "x", "verb": "grasp", "split": "train", "vertex_semantics_path": "f/sem.pt"}\n',
        encoding="utf-8",
    )
    ds = DataRootDataset(data_root=root, manifest_path=manifest, cfg={}, split="train")
    item = ds[0]
    assert item["vertex_features"].shape == (5, 16)
    assert item["vertex_visible_mask"].dtype == torch.bool


def test_vertex_training_epoch_smoke(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "meshes").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "feat").mkdir(parents=True)
    (root / "lat").mkdir(parents=True)
    mesh_path = root / "meshes" / "x.obj"
    trimesh.creation.box().export(mesh_path)
    np.save(root / "labels" / "x.npy", np.linspace(0, 1, 8, dtype=np.float32))
    torch.save(
        {
            "features": torch.randn(8, 16),
            "view_counts": torch.ones(8),
            "visible_in_any_view": torch.ones(8, dtype=torch.bool),
        },
        root / "feat" / "sem.pt",
    )
    torch.save({"global_latent": torch.zeros(4)}, root / "lat" / "g.pt")

    manifest = root / "manifest.jsonl"
    manifest.write_text(
        '{"sample_id": "a", "verb": "grasp", "split": "train", "mesh_path": "meshes/x.obj", '
        '"vertex_affordance_path": "labels/x.npy", "vertex_semantics_path": "feat/sem.pt", '
        '"sam3d_global_latent_path": "lat/g.pt"}\n',
        encoding="utf-8",
    )

    cfg = {"model": {"vlm_dim": 16, "verb_dim": 16, "num_verbs": 1, "sam3d_dim": 0, "hidden_dims": [32], "dropout": 0.0}}
    ds = DataRootDataset(data_root=root, manifest_path=manifest, cfg=cfg, split="train")
    model = AffordanceMLP(MLPHeadConfig(vlm_dim=16, verb_dim=16, num_verbs=1, sam3d_dim=0, hidden_dims=(32,), dropout=0.0))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    verb_to_idx = {"grasp": 0}
    device = torch.device("cpu")
    loss0 = training_epoch_vertex_bce(model, opt, ds, verb_to_idx=verb_to_idx, device=device)
    loss1 = training_epoch_vertex_bce(model, opt, ds, verb_to_idx=verb_to_idx, device=device)
    assert loss0 > 0 and loss1 > 0
    ev = eval_vertex_bce(model, ds, verb_to_idx=verb_to_idx, device=device)
    assert ev == ev
