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

## Development without SAM3D checkpoints

You can work on rendering, VLM features, and projection using a mesh only:

- Asset: `data/sample.glb` (no paired Gaussian splat required)
- Set `rendering.backend: mesh` in `configs/default.yaml`
- Affordance labels and projection always use mesh vertices

When SAM3D is available, each sample may include `mesh.glb` and `gaussian.ply`; set `backend` to `gaussian` or `both` to use splat rendering for novel views.

## SAM3D batch reconstruction

```bash
python dataset_pipeline.py \
  --dataset_dir /path/to/dataset \
  --output_dir outputs/reconstructions
```

## 3D Gaussian splat → multi-view RGB (SAM3D-style inputs)

From a **3DGS `.ply`** (Inria / Nerfstudio layout), render an orbit of RGB + depth:

- **True splat (CUDA + `gsplat`):** `pip install -e ".[gsplat]"` (or SAM3D Docker `[inference]`), then:

```bash
PYTHONPATH=src python scripts/render_gaussian_views.py \
  --backend gsplat \
  --splat_path examples/gaussian_splat/tiny_gaussians.ply \
  --output_dir outputs/gaussian_renders/demo
```

- **Centre preview only (pyrender):** `--backend preview` (no `gsplat`; draws Gaussian means as points/icospheres).

Visual walkthrough: `notebooks/03_rendering_gaussian_splat.ipynb` (**gsplat** RGB → **`exports/gaussian_splat/`**; set **`SPLAT_PLY`** in the notebook). Details: [docs/data_layout.md](docs/data_layout.md).

## Stage testing notebooks

Each pipeline stage has a required debug notebook under `notebooks/`. See [notebooks/README.md](notebooks/README.md) and [implementation_order.md](docs/implementation_order.md#notebook--stage-map-current-status) for **what you can work on now**.

| Next up (no SAM3D checkpoints) | Blocked |
|----------------------------------|---------|
| `06_affordance_head_debug.ipynb` | SAM3D run (`01_…`), full training (`07_…` needs **3DAffordSplat** pipeline) |

Steps **0–5** implemented. Run notebooks `00` → `02` → `04` → `05` in order (add `03` for gsplat, `06` for affordance head).

```bash
jupyter lab notebooks/00_mesh_bootstrap.ipynb
```

## Package layout (MVP)

```
src/
├── reconstruction/   # SAM3D wrapper, mesh I/O
├── rendering/
├── vlm/
├── projection/
├── models/
├── datasets/         # mesh loading; 3DAffordSplat / AffordSplat (planned)
├── training/
├── visualization/
└── utils/
notebooks/            # 00–08 per-stage validation (required)
```
