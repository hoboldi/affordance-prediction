# Affordance Prediction

Open-vocabulary affordance prediction by combining **SAM3D** reconstruction with frozen **VLM**
semantic features, supervised by **affordance pseudolabels from the frozen [GEAL](https://github.com/DylanOrange/geal) teacher**.

**Pipeline:** a clean single-object image (from **OmniObject3D**) → SAM3D reconstructs a mesh →
frozen VLM patch features projected onto mesh vertices → small per-vertex MLP head. Supervision comes
from running GEAL on the *same* reconstructed geometry, so labels are already in mesh-vertex order —
**no cross-modal alignment is needed**. (This replaces the earlier 3DAffordSplat + ICP-alignment
approach, which failed because SAM3D reconstructs splat renders poorly.)

See [docs/project.md](docs/project.md) for the full pipeline, [docs/data_strategy_geal_omniobject3d.md](docs/data_strategy_geal_omniobject3d.md)
for the data + supervision plan, and [docs/implementation_order.md](docs/implementation_order.md) for the MVP development order.

## Fresh server (start here)

Recommended transfer is **`git clone` of the `geal-pseudolabels` branch** (not copying the working dir —
`external/` and `data/` are gitignored). Then run the bootstrap, which restores everything that doesn't
travel with git (SAM3D submodule, GEAL code, optional GEAL weights) and prints the remaining steps:

```bash
git clone -b geal-pseudolabels git@github.com:hoboldi/affordance-prediction.git
cd affordance-prediction
bash scripts/bootstrap.sh            # SAM3D submodule + GEAL code (light)
bash scripts/bootstrap.sh --weights  # also fetch GEAL checkpoints (needs storage)
```

Then create the env and follow [Data + supervision pipeline](#data--supervision-pipeline) below.

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

### SAM3D (reconstruction backend)

```bash
git submodule update --init sam-3d-objects
```

Checkpoints and extra CUDA deps: see the `sam-3d-objects` submodule.
`ModuleNotFoundError: No module named 'inference'` means Python cannot see **`sam-3d-objects/notebook/inference.py`** (submodule not inited, wrong `SAM3D_OBJECTS_ROOT`, or an **empty** `sam-3d-objects/` overwriting the Docker image's clone).

### GEAL teacher (pseudolabels)

GEAL is used as a frozen teacher. Its **code** is cloned into `external/geal` (gitignored); its
**weights** are fetched separately.

```bash
# code only (small) — already present if you followed setup:
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/DylanOrange/geal external/geal

# weights (later; ~hundreds of MB) → external/geal/ckpt/
#   from https://huggingface.co/datasets/dylanorange/geal : piad_seen.pt / piad_unseen.pt / laso_*.pt
```

Only GEAL's **3D branch** (`model.branch_3d.Branch3D`) is used at inference — a PointNet++ + RoBERTa
model that maps a point cloud + affordance question to per-point scores in `[0, 1]`. The Gaussian-splatting
2D branch is training-only, so **no CUDA rasterizer build is needed** for pseudolabeling.

### Docker (SAM3D + full pipeline on NVIDIA GPU)

```bash
docker compose build
docker compose run --rm autonomous-pipeline bash
```

Details, VRAM expectations, Jupyter, Hugging Face checkpoints: [docker/README.md](docker/README.md).

## Data + supervision pipeline

End-to-end (the data/weights steps are deferred until disk + GPU are available):

```bash
# 1. Stage OmniObject3D images for the GEAL-overlapping categories + write a manifest
PYTHONPATH=src python scripts/prepare_omniobject3d.py --config configs/omniobject3d.yaml

# 2. Reconstruct each staged image with SAM3D  (mesh.glb + SLAT latents per object)
PYTHONPATH=src python scripts/generate_sam3d.py \
  --dataset_dir data/omniobject3d/sam3d_inputs \
  --output_dir  data/omniobject3d/reconstructions

# 3. Generate per-vertex affordance pseudolabels with GEAL
PYTHONPATH=src python scripts/generate_geal_pseudolabels.py \
  --manifest data/omniobject3d/manifest.jsonl \
  --ckpt external/geal/ckpt/piad_seen.pt

# 4. Train the per-vertex affordance head on the pseudolabeled manifest
PYTHONPATH=src python scripts/train_affordance.py \
  --manifest data/omniobject3d/manifest.pseudolabeled.jsonl \
  --output_dir outputs/affordance_mlp
```

GEAL only knows 3D-AffordanceNet's **23 object / 18 affordance** classes, so OmniObject3D is filtered
to overlapping categories via `category_map` in [configs/omniobject3d.yaml](configs/omniobject3d.yaml)
(extend it against the actual downloaded folder names). Each kept class is queried with one canonical
affordance (`affordance_map`).

### On-disk dataset (`data/`)

Training samples are listed in a **`manifest.jsonl`** (default `data/omniobject3d/manifest.jsonl`, or
`AFFORDANCE_DATA_ROOT`). Each row carries `sample_id`, `verb` (affordance), `object_class` (GEAL class),
`sam3d_reconstruction_dir`, and `vertex_pseudolabel_path` (a `(V,)` tensor in mesh-vertex order). See
[docs/data_layout.md](docs/data_layout.md) and `datasets.DataRootDataset`.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torchvision::nms does not exist` | `pip uninstall -y torch torchvision torchaudio` then `pip install -r requirements-pytorch.txt` |
| `No module named 'inference'` (SAM3D) | `git submodule update --init sam-3d-objects`, or set `SAM3D_OBJECTS_ROOT` |
| `checkpoints/hf/pipeline.yaml` missing (SAM3D) | Weights on Hugging Face ([facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)); run `sam-3d-objects/doc/setup.md` §2 |
| GEAL `checkpoint not found` | Download weights into `external/geal/ckpt/` from [dylanorange/geal](https://huggingface.co/datasets/dylanorange/geal) |

## Development without checkpoints

You can work on rendering, VLM features, and projection using a mesh only:

- Asset: `data/sample.glb`
- Set `rendering.backend: mesh` or `rendering.splat_path: null` in `configs/default.yaml` for **mesh-colour** RGB without a paired `.ply`.
- Affordance labels and projection always use mesh vertices.

When SAM3D is available, each sample includes `mesh.glb` and usually `gaussian.ply`; `backend: gaussian`
uses **splat RGB** with **mesh** depth and vertex correspondences.

## Stage testing notebooks

Each pipeline stage has a debug notebook under `notebooks/debug/`. See [notebooks/README.md](notebooks/README.md).
Run `00` → `02` → `04` → `05` → `06` in order (add `03` for gsplat rendering of SAM3D output).

```bash
jupyter lab notebooks/debug/00_mesh_bootstrap.ipynb
```

## Package layout

```
src/
├── reconstruction/   # SAM3D wrapper, mesh I/O
├── rendering/        # mesh + gaussian-splat (gsplat) novel-view rendering
├── vlm/              # frozen CLIP patch + text features
├── projection/       # 2D patch features → mesh vertices
├── models/           # AffordanceMLP (mlp_head.py)
├── datasets/         # DataRootDataset, OmniObject3D ingestion
├── labeling/         # GEAL teacher wrapper (geal_infer.py)
├── training/
└── utils/
scripts/              # prepare_omniobject3d, generate_sam3d, generate_geal_pseudolabels, train_affordance
external/geal/        # GEAL clone (gitignored; code only — weights fetched separately)
```
