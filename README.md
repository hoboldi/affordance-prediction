# ReVerb — Verb-Conditioned Affordance Prediction on Single-Image 3D Reconstructions

ReVerb predicts, for a given **action verb**, *where on a 3D object that action applies* — a
per-vertex affordance map on geometry **reconstructed from a single RGB image**. It removes the
usual dependence on a sensed point cloud or scan: an object is reconstructed with
[SAM3D](https://github.com/facebookresearch/sam-3d-objects), and a verb-conditioned spatial GNN
grounds the verb onto the reconstructed mesh.

The same object routes to **different regions for different verbs** — a cup is *grasped* at the
handle, *poured* from the rim, *contained* in the bowl.

## Method

```
single RGB image ──SAM3D──▶ mesh ──▶ per-vertex features ──▶ verb-conditioned GNN ──▶ affordance map
                                        │                          ▲
                                        ├─ DINOv2 appearance (128-d, back-projected multi-view)
                                        ├─ local geometry (5 descriptors)
                                        └─ CLIP-text embedding of the verb ────────────┘
```

- **Per-vertex representation.** DINOv2-base patch tokens rendered from multiple views and
  back-projected to the surface (128-d after a fixed random projection), five interpretable local
  geometric descriptors, and a CLIP-text embedding of the verb — concatenated to a 261-d input.
- **Head.** An EdgeConv spatial GNN (k=24 kNN graph, 3 layers, width 128) with the verb injected at
  every layer, so message passing is verb-dependent throughout. Neighbourhood aggregation yields
  spatially coherent maps a per-vertex MLP cannot.
- **Training.** Two stages: (1) **distill** a frozen GEAL teacher's per-vertex pseudo-labels on the
  reconstructed meshes; (2) **finetune** on 215 human-labeled objects with class-balanced BCE.
  Evaluation is 5-fold cross-validation over objects (macro AUPRC).

CLIP-vision, SAM3D structured latents, and surface normals were evaluated and dropped — each adds
≈0 AUPRC. CLIP enters the model only as the **text** encoder for the verb.

## Key results

- **0.86 macro AUPRC** under 5-fold CV on 215 human-annotated objects (8 verbs, 10 CO3D categories).
- Affordance verbs split into two classes: **appearance** verbs (*grasp, display, press*) depend on
  learned DINOv2 features; **geometric** verbs (*contain, pour, sit, move, lift*) do not. The split is
  statistically robust (bootstrap CIs) and survives region-size controls and teacher removal.
- **Cross-category transfer follows the split:** geometric verbs transfer to unseen object categories
  (e.g. *pour* ≈0.82), appearance verbs do not (*grasp* 0.15–0.51).

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full result table and analyses, and
[`notebooks/reverb_demo.ipynb`](notebooks/reverb_demo.ipynb) for a runnable demo of verb routing.

## Setup

`sam-3d-objects/` is a git submodule. After cloning, restore it and create the environment:

```bash
git submodule update --init                 # SAM3D
bash scripts/bootstrap.sh                    # SAM3D + GEAL teacher code
conda env create -f environment.yml          # or: pip install -r requirements.txt
```

All scripts assume the repo root as the working directory with `PYTHONPATH=src`.

## Pipeline

Data (reconstructions and per-vertex features) live outside the repo; see
[`docs/data_layout.md`](docs/data_layout.md) for the on-disk layout. To reproduce end to end:

| Step | Script |
|---|---|
| Prepare CO3D objects | `scripts/prepare_co3d.py` |
| Reconstruct meshes (SAM3D) | `scripts/generate_sam3d.py` |
| DINOv2 per-vertex features | `scripts/generate_vertex_dino.py` |
| Geometry descriptors | `scripts/precompute_geom.py` |
| GEAL teacher pseudo-labels | `scripts/generate_geal_pseudolabels.py` |
| Stage 1 — distillation pretrain | `experiments/cv_harness/train_gnn_pretrain.py` |
| Stage 2 — 5-fold finetune on human labels | `experiments/cv_harness/train_gnn_cv.py` |
| Evaluation / generalization | `scripts/eval_human_gt.py`, `scripts/eval_generalization.py` |

## Repository layout

```
src/                 model, dataset, projection, rendering, VLM wrappers
scripts/             data-generation, training, evaluation, and rendering scripts
experiments/         5-fold CV harness (cv_harness/) + analysis scripts
notebooks/           reverb_demo.ipynb — runnable verb-routing demo
final_figures/       curated figures
docs/                data_layout.md, RESULTS.md
tests/               unit tests
configs/             rendering / pipeline configuration
examples/            tiny fixtures illustrating the data format
```

## Acknowledgements

Builds on [SAM3D](https://github.com/facebookresearch/sam-3d-objects) for single-image
reconstruction, [GEAL](https://github.com/DylanOrange/geal) as the distillation teacher, DINOv2 for
appearance features, and CLIP for verb-text embeddings. Objects are drawn from
[CO3D](https://github.com/facebookresearch/co3d).
