Minimal on-disk example for [`DataRootDataset`](../../src/datasets/data_root_dataset.py).

A minimal **train/val** manifest with `vertex_semantics_path` lives at [`manifest_training.jsonl`](manifest_training.jsonl); run **`python scripts/build_example_training_fixtures.py`** once so those paths resolve. The default [`manifest.jsonl`](manifest.jsonl) stays a **single-row** mesh+labels example for unit tests without torch fixtures.

```bash
python scripts/build_example_training_fixtures.py
```

From the repository root, with `PYTHONPATH=src`:

```python
from pathlib import Path
from datasets.data_root_dataset import DataRootDataset

root = Path("examples/data_manifest")
ds = DataRootDataset(
    data_root=root,
    manifest_path=root / "manifest_training.jsonl",
    cfg={},
    split="train",
    load_mesh_eager=True,
    load_vertex_labels_eager=True,
)
print(ds[0]["sample_id"], ds[0]["vertex_affordance"].shape)
print("vertex_features", ds[0]["vertex_features"].shape)
```

Use the same layout under your real `data/` directory (or set `AFFORDANCE_DATA_ROOT` to e.g. `/workspace/data`).