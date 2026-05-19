# Affordance Prediction

Open-vocabulary affordance prediction by combining SAM3D reconstruction with frozen VLM semantic features.

See [docs/project.md](docs/project.md) for the full pipeline and [docs/implementation_order.md](docs/implementation_order.md) for the MVP development plan.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
git submodule update --init sam-3d-objects
```

SAM3D checkpoints and extra dependencies are documented in the `sam-3d-objects` submodule.

## SAM3D batch reconstruction

```bash
python dataset_pipeline.py \
  --dataset_dir /path/to/dataset \
  --output_dir outputs/reconstructions
```

## Package layout (MVP)

```
src/
├── reconstruction/   # SAM3D wrapper, mesh I/O
├── rendering/
├── vlm/
├── projection/
├── models/
├── datasets/         # mesh loading, AGD20K (later)
├── training/
├── visualization/
└── utils/
```
