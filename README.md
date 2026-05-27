# Affordance Prediction

Open-vocabulary affordance prediction by combining SAM3D reconstruction with frozen VLM semantic features.

See [docs/project.md](docs/project.md) for the full pipeline, [docs/data_strategy_3daffordsplat.md](docs/data_strategy_3daffordsplat.md) for the **3DAffordSplat** data plan (synthetic views → SAM3D), and [docs/implementation_order.md](docs/implementation_order.md) for the MVP development plan.

## Setup

**Python 3.10–3.11 recommended.** Install **torch and torchvision as a matched pair** (≥2.6 / ≥0.21) before the rest — do not mix conda `pytorch` with pip `torchvision`.

### Conda (recommended)

```bash
cd affordance-prediction
conda env create -f environment.yml
conda activate affordance
python -m ipykernel install --user --name affordance --display-name "affordance (conda)"
```

Or manually:

```bash
conda create -n affordance python=3.11 -y
conda activate affordance
conda install -c conda-forge libjpeg-turbo jpeg -y   # optional; reduces torchvision libjpeg warnings on macOS

pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install -r requirements-pytorch.txt
pip install -r requirements.txt
pip install -e ".[notebooks,dev]"
```

Verify:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
PYTHONPATH=src python -c "from vlm.vlm_wrapper import VLMWrapper; VLMWrapper(); print('ok')"
```

### venv (alternative)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pytorch.txt
pip install -e ".[notebooks,dev]"
```

### SAM3D (optional, later)

```bash
git submodule update --init sam-3d-objects
```

Checkpoints and extra CUDA deps: see the `sam-3d-objects` submodule.

`ModuleNotFoundError: No module named 'inference'` means Python cannot see **`sam-3d-objects/notebook/inference.py`** (submodule not inited, wrong `SAM3D_OBJECTS_ROOT`, or an **empty** `sam-3d-objects/` on the host **overwriting** the Docker image’s clone). It is not fixed by `pip install gsplat` alone.

### Docker (SAM3D + full pipeline on NVIDIA GPU)

Use this when you want a Linux + CUDA 12.1 environment aligned with Meta’s SAM3D install without touching your host Python. The image **clones `sam-3d-objects` during the build** (no submodule required). If you use a submodule locally, the repo bind mount still replaces `/workspace/sam-3d-objects` at runtime.

```bash
docker compose build
docker compose run --rm sam3d-pipeline bash
```

Details, VRAM expectations, Jupyter, Hugging Face checkpoints, and optional `SAM3D_REF` / `SAM3D_REPO` build args: [docker/README.md](docker/README.md).

### On-disk dataset (`data/`)

Training samples are listed in **`data/manifest.jsonl`** (or `AFFORDANCE_DATA_ROOT`, e.g. `/workspace/data`). See [docs/data_layout.md](docs/data_layout.md) and `datasets.DataRootDataset`. For a **local AffordSplat / 3DAffordSplat** tree (e.g. under **`/data/Seen/...`**), use `datasets.AffordSplatLocalDataset` and `AFFORDANCE_AFFORDSPLAT_ROOT`.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torchvision::nms does not exist` | `pip uninstall -y torch torchvision torchaudio` then `pip install -r requirements-pytorch.txt` |
| `torch.load` / need torch ≥2.6 | Same as above; CLIP uses `use_safetensors=True` but transformers still imports torchvision |
| `libjpeg.9.dylib` warning (macOS) | `conda install -c conda-forge libjpeg-turbo jpeg -y` |
| `zsh: 0.4 not found` on pip install | Quote specs: `pip install "safetensors>=0.4"` |
| `No module named 'inference'` (SAM3D) | `git submodule update --init sam-3d-objects`, or set `SAM3D_OBJECTS_ROOT`; in Docker, remove an empty host `sam-3d-objects/` so the image clone is visible |
| `checkpoints/hf/pipeline.yaml` missing (SAM3D) | Weights are on Hugging Face ([facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)); after access, run `sam-3d-objects/doc/setup.md` §2 (`hf download` …) inside `sam-3d-objects/`; use `HF_TOKEN` or `hf auth login` |

## Development without SAM3D checkpoints

You can work on rendering, VLM features, and projection using a mesh only:

- Asset: `data/sample.glb`
- Set `rendering.backend: mesh` or `rendering.splat_path: null` in `configs/default.yaml` if you want **mesh-colour** RGB without a paired `.ply` (the repo default is `backend: gaussian` with `examples/gaussian_splat/tiny_gaussians.ply` for smoke tests — appearance may not match `sample.glb` until you use a SAM3D `reconstruction/` with aligned `mesh.glb` + `gaussian.ply`).
- Affordance labels and projection always use mesh vertices

When SAM3D is available, each sample includes `mesh.glb` and usually `gaussian.ply`; `cfg_with_sam3d_reconstruction` sets `splat_path` and `backend: gaussian` so **splat RGB** is used with **mesh** depth and vertex correspondences.

## SAM3D batch reconstruction

```bash
python dataset_pipeline.py \
  --dataset_dir /path/to/dataset \
  --output_dir outputs/reconstructions
```

## 3D Gaussian splat → multi-view RGB (SAM3D-style inputs)

From a **3DGS `.ply`** (Inria / Nerfstudio layout), render an **azimuth-only** orbit of RGB + depth at a fixed pitch in **`[elevation_min_deg, elevation_max_deg]`** intersected with the project band **`[25°, 60°]`** above the ground plane (no horizon-grazing; see `configs/default.yaml` → `rendering`):

- **True splat (CUDA + `gsplat`):** `pip install -e ".[gsplat]"` (or SAM3D Docker `[inference]`), then:

```bash
PYTHONPATH=src python scripts/render_gaussian_views.py \
  --backend gsplat \
  --splat_path examples/gaussian_splat/tiny_gaussians.ply \
  --output_dir outputs/gaussian_renders/demo
```

**Best single RGB vs GT point cloud:** score orbit views with depth unprojection + k-NN alignment, export one PNG for MLLM / SAM3D (CUDA + `gsplat`; see [docs/gsplat_gt_view_selection.md](docs/gsplat_gt_view_selection.md)):

```bash
PYTHONPATH=src python scripts/select_best_gsplat_view_for_gt.py \
  --splat_path examples/gaussian_splat/tiny_gaussians.ply \
  --gt_path path/to/gt.ply \
  --output_png exports/best_view/best_rgb.png
```

Visual walkthrough (all candidates + winner): **`notebooks/11_gsplat_view_selection_vs_gt.ipynb`**.

**gsplat → SAM3D (mesh + cached latent):** rasterise the same `.ply`, then run SAM3D on `view_000` (see [docs/pipeline_gsplat_sam3d.md](docs/pipeline_gsplat_sam3d.md), `notebooks/10_sam3d_from_gsplat.ipynb`). **You cannot use this path outside the `sam3d-pipeline` Docker container** in a supported way — run inside `docker compose run --rm sam3d-pipeline bash` (see [docker/README.md](docker/README.md)):

```bash
docker compose run --rm sam3d-pipeline bash -lc 'cd /workspace && PYTHONPATH=src python scripts/render_gsplat_and_sam3d.py --splat_path data/Seen/train/bag/Gaussian/GS_0017.ply --run_dir exports/gsplat_sam3d/my_run'
```

**Batch / idempotent SAM3D** (same on-disk layout; skips objects that already have `mesh.glb` + matching `meta_prerender.json`): run `PYTHONPATH=src python scripts/batch_gsplat_sam3d.py` inside the same container — see [docs/pipeline_gsplat_sam3d.md](docs/pipeline_gsplat_sam3d.md) for flags (`--splat_paths_file`, `--affordsplat_random_n`, …).

- **Centre preview only (pyrender):** `--backend preview` (no `gsplat`; draws Gaussian means as points/icospheres).

Visual walkthrough: `notebooks/03_rendering_gaussian_splat.ipynb` (**gsplat** RGB → **`exports/gaussian_splat/`**; set **`SPLAT_PLY`** in the notebook). SAM3D from gsplat: **`notebooks/10_sam3d_from_gsplat.ipynb`**. Details: [docs/data_layout.md](docs/data_layout.md), [docs/pipeline_gsplat_sam3d.md](docs/pipeline_gsplat_sam3d.md).

## Stage testing notebooks

Each pipeline stage has a required debug notebook under `notebooks/`. See [notebooks/README.md](notebooks/README.md) and [implementation_order.md](docs/implementation_order.md#notebook--stage-map-current-status) for **what you can work on now**.

| Next up (no SAM3D checkpoints) | Blocked |
|----------------------------------|---------|
| `06_affordance_head_debug.ipynb` | Single-mesh head debug (`05` caches); **`07_training_evaluation_debug.ipynb`** uses **manifest** + optional `vertex_semantics_path` (toy `examples/data_manifest` + fixture script) |

Steps **0–7** scaffolding is in place. Run notebooks `00` → `02` → `04` → `05` → `06` → `07` in order (add `03` for gsplat, **`10`** before **`02`** when using SAM3D meshes + `manifest_training` / fixture script for **07**).

```bash
jupyter lab notebooks/00_mesh_bootstrap.ipynb
```

## Package layout (MVP)

```
src/
├── reconstruction/   # SAM3D wrapper, mesh I/O, gsplat→SAM3D (`gsplat_to_sam3d.py`)
├── rendering/
├── vlm/
├── projection/
├── models/           # AffordanceMLP (`mlp_head.py`)
├── datasets/         # DataRootDataset, AffordSplatLocalDataset
├── training/         # `affordance_fit.fit_affordance_mlp_simple`
├── visualization/
└── utils/
notebooks/            # 00–08 + 10 per-stage validation
```
