# On-disk data layout (`data/`)

The default dataloader ([`DataRootDataset`](../src/datasets/data_root_dataset.py)) reads **`manifest.jsonl`** under the **data root** (`paths.data_root` in `configs/default.yaml`, usually `data/` → `/workspace/data` when the project root is `/workspace`).

Override the root at runtime with:

```bash
export AFFORDANCE_DATA_ROOT=/workspace/data
```

## Manifest format (`manifest.jsonl`)

One JSON object per line (JSONL). Paths are **relative to the data root** unless absolute.

| Field | Required | Description |
|--------|----------|-------------|
| `sample_id` | yes | Stable string ID |
| `verb` | no | Open-vocabulary verb for conditioning (default `grasp`) |
| `mesh_path` | no | Mesh for rendering / projection (e.g. `meshes/foo.glb`) |
| `splat_path` | no | Gaussian splat `.ply` if used as render backend |
| `reference_rgb_path` | no | RGB image for SAM3D-style reconstruction |
| `mask_path` | no | Foreground / instance mask aligned with reference RGB |
| `vertex_affordance_path` | no | Per-vertex supervision: `.npy` (float32 vector) or `.pt` (tensor or dict with `vertex_affordance`) |
| `vertex_semantics_path` | no | Per-vertex VLM features (``.pt`` with ``features``, ``visible_in_any_view`` — same as notebook **05** ``vertex_semantic.pt``) |
| `sam3d_global_latent_path` | no | Cached `(8,)` latent from `reconstruction.sam3d_wrapper` (`.pt` with `global_latent` key or raw tensor) |
| `split` | no | `train` / `val` / … — filter with `DataRootDataset(..., split="train")` |

Any other keys are passed through as `extras` on each sample dict.

## Example line

```json
{"sample_id": "chair_01", "verb": "sit on", "split": "train", "mesh_path": "meshes/chair_01.glb", "vertex_affordance_path": "labels/chair_01.npy", "vertex_semantics_path": "cache/vertex_sem/chair_01.pt", "sam3d_global_latent_path": "cache/sam3d/chair_01/global_latent.pt"}
```

A tracked **minimal** example lives at [`examples/data_manifest/manifest.jsonl`](../examples/data_manifest/manifest.jsonl) (mesh + labels only). For **notebook 07**, use [`manifest_training.jsonl`](../examples/data_manifest/manifest_training.jsonl) after running `scripts/build_example_training_fixtures.py`.

## Recommended directories

```text
data/
  manifest.jsonl
  meshes/           # .glb / .obj / …
  splats/           # optional .ply
  renders/          # optional multi-view RGB from 3DGS
  labels/           # vertex affordance .npy / .pt
  cache/sam3d/      # SAM3D global latents (see configs/default.yaml paths.cache_root)
```

Large binaries and `*.pt` caches are typically gitignored; keep them on disk or on a mounted volume (e.g. Docker `/data/inNet` → `data/`).

## Gaussian splat → 2D views (SAM3D inputs)

Use `scripts/render_gaussian_views.py` to rasterise a **3DGS `.ply`** into orbit RGB/depth (point-centre preview; see `rendering/gaussian_point_renderer.py`). Typical output:

```text
outputs/gaussian_renders/<run_id>/
  view_000.png
  view_000_depth.npy
  ...
  cameras.json
```

Point `DataRootDataset` manifest fields (`reference_rgb_path`, …) at these paths once you wire training.

## 3DAffordSplat / AffordSplat local mirror (no manifest)

If you already have the **Hugging Face dataset tree** on disk (e.g. cloned or copied under **`/data`**), use [`AffordSplatLocalDataset`](../src/datasets/affordsplat_local_dataset.py). It walks **`…/Seen/<train|val|test>/<category>/Gaussian/GS_*.ply`** and pairs **`GS_anno_<id>.ply`** from sibling affordance folders (`grasp`, `contain`, … — not `Gaussian` / `PointCloud`).

**Root directory** (must contain a `Seen/` folder, or use `subset=` for other top-level splits you keep):

```bash
export AFFORDANCE_AFFORDSPLAT_ROOT=/data
```

Or set `paths.affordsplat_root` in YAML (repo-relative or absolute). If neither is set, the loader tries **`AFFORDANCE_DATA_ROOT`** / config **`data_root`** when **`<that>/Seen`** exists, then **`/workspace/data`**, then **`/data`** when **`/data/Seen`** exists.

Walkthrough: [`notebooks/01_affordsplat_dataloader.ipynb`](../notebooks/01_affordsplat_dataloader.ipynb) (zero-config when `/data/Seen` exists).

```python
from pathlib import Path
from datasets import AffordSplatLocalDataset

# Explicit root if your tree is not exactly /data/Seen (e.g. /data/Weizm/AffordSplat)
ds = AffordSplatLocalDataset(affordsplat_root=Path("/data"), subset="Seen", split="train")
sample = ds[0]
print(sample["sample_id"], sample["verb"], sample["splat_path"])
# extras: affordsplat_category, affordsplat_gs_anno_path (when present), …
```

Each `__getitem__` dict matches `DataRootDataset` keys where applicable (`sample_id`, `verb`, `split`, `splat_path`, `extras`, …) so you can reuse the same training loop entry point.
