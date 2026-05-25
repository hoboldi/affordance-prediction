Minimal on-disk example for [`DataRootDataset`](../../src/datasets/data_root_dataset.py).

From the repository root, with `PYTHONPATH=src`:

```python
from pathlib import Path
from datasets.data_root_dataset import DataRootDataset

root = Path("examples/data_manifest")
ds = DataRootDataset(
    data_root=root,
    manifest_path=root / "manifest.jsonl",
    cfg={},
    load_mesh_eager=True,
    load_vertex_labels_eager=True,
)
print(ds[0]["sample_id"], ds[0]["vertex_affordance"].shape, ds[0]["mesh"].num_vertices)
```

Use the same layout under your real `data/` directory (or set `AFFORDANCE_DATA_ROOT` to e.g. `/workspace/data`).