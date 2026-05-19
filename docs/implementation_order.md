# Stage 0 — Minimal Research Goal

The first prototype should answer only one question:

> “Can VLM semantic features projected onto a reconstructed 3D mesh help predict verb-conditioned affordances?”

Everything else should initially be simplified.

That means:

* no transformers,
* no advanced visibility reasoning,
* no learned view selection,
* no graph networks,
* no uncertainty modeling,
* no splat-native semantic fusion (splats may still be used as a render backend),
* no complex losses.

The first version should simply prove:

1. 3D reconstruction works,
2. VLM patch features can be projected onto vertices,
3. concatenated features predict affordances.

---

# Phase 1 — Absolute Minimal Prototype (MVP)

## Objective

Build a working end-to-end pipeline with:

* one reconstruction,
* one VLM,
* one projection method,
* one MLP head.

---

# MVP Pipeline

```text
RGB-D image + verb
        ↓
SAM3D reconstruction (or preloaded mesh.glb)
        ↓
Mesh (+ optional gaussian.ply)
        ↓
Render 4–8 novel views (mesh and/or splat backend)
        ↓
Frozen VLM patch embeddings
        ↓
Project patches onto mesh vertices
        ↓
Concatenate:
[SAM3D latent | VLM latent | verb embedding]
        ↓
Small MLP
        ↓
Per-vertex affordance score
```

---

# MVP Components

## 1. Reconstruction

### Keep:

* mesh output (required for affordances and projection)

### Optional:

* `gaussian.ply` when SAM3D checkpoints are available (enables splat rendering)

### Ignore for now:

* PointNet global features
* splat-based semantic accumulation (Phase 2+)

### Goal:

Stable mesh vertices and correspondences. Splats are not required for MVP.

---

## 2. Rendering

### Keep:

* fixed spherical camera sampling
* 4–8 views
* **mesh renderer** (always — works with `data/sample.glb` alone)
* **gaussian splat renderer** when a `.ply` exists next to the mesh

### Config:

```yaml
rendering:
  backend: mesh          # mesh | gaussian | both
  mesh_path: data/sample.glb
  splat_path: null       # optional; e.g. outputs/.../gaussian.ply
```

### Ignore:

* visibility confidence weighting
* adaptive view selection
* comparing mesh vs splat in training (debug/ablation only)

### Goal:

RGB + depth + vertex correspondences for VLM and projection. Mesh-only is enough until SAM3D outputs splats.

---

## 3. VLM

### Keep:

* frozen model
* final-layer patch embeddings

### Ignore:

* intermediate layers
* attention maps
* cross-attention

### Goal:

Obtain stable semantic features quickly.

---

## 4. Projection

### Keep:

* nearest visible patch assignment
* simple averaging across views

### Ignore:

* learned fusion
* attention pooling
* uncertainty

### Goal:

Validate semantic grounding onto geometry.

---

## 5. Feature Fusion

### Keep:

```python
vertex_feature = concat(
    sam3d_feature,
    vlm_feature,
    verb_embedding
)
```

### Ignore:

* PointNet globals
* transformers
* graph reasoning

---

## 6. Prediction Head

### Keep:

* 2–3 layer MLP

### Ignore:

* transformers
* GNNs

---

## 7. Training

### Keep:

* BCE loss
* AGD20K supervision

### Ignore:

* consistency losses
* smoothness
* contrastive alignment

---

# Minimal Repository Structure

The MVP can be dramatically smaller.

```text
src/
├── reconstruction/
├── rendering/
├── vlm/
├── projection/
├── models/
├── datasets/
├── training/
└── visualization/
notebooks/          # required — one debug notebook per pipeline stage (see below)
```

You do not yet need:

* experiments/
* graph models/
* transformer heads/
* uncertainty/
* advanced evaluation.

---

# Stage testing notebooks (required)

Each pipeline stage must have a **dedicated Jupyter notebook** used to validate that stage in isolation before wiring the full training loop. Notebooks are the primary integration test for geometry and visuals; `tests/` covers unit-level logic.

## Requirements (all stage notebooks)

* Load settings from `configs/default.yaml` (paths, `rendering.backend`, etc.).
* State **prerequisites** in the first cell (e.g. “needs `data/sample.glb` only” vs “needs SAM3D checkpoints”).
* Produce **inline figures** (mesh views, render grids, heatmaps) — not only saved files.
* End with an explicit **pass checklist** (assertions or printed criteria); fail loudly if outputs are empty or degenerate.
* Write optional caches under `outputs/notebooks/<stage>/` so reruns are fast.
* Prefer the same APIs as `src/` (import from installed package or `PYTHONPATH=src`).

## Notebook ↔ stage map (current status)

**Legend:** ✅ done · 🟡 in progress / partial · ⬜ not started · 🚫 blocked

| Step | Notebook | `src/` code | Status | Can work on now? | Blocker / notes |
|------|----------|-------------|--------|------------------|-----------------|
| 0 | `00_mesh_bootstrap.ipynb` | `datasets/mesh_loading.py`, `utils/config.py` | ✅ | — | Run notebook to confirm on your machine |
| 1 | `01_reconstruction_debug.ipynb` | `reconstruction/sam3d_wrapper.py`, `mesh_utils.py` | 🟡 | **No** (run) | Code exists; **SAM3D checkpoints** + submodule setup required to execute |
| 2 | `02_rendering_debug.ipynb` | `rendering/mesh_renderer.py`, `camera_sampling.py` | ✅ | — | Mesh backend; run notebook to confirm |
| 3 | `03_vlm_features_debug.ipynb` | `vlm/vlm_wrapper.py`, `patch_extractor.py`, `text_encoder.py` | ✅ | — | Frozen CLIP-B/32; run notebook to confirm |
| 4 | `04_projection_debug.ipynb` | `projection/project_to_mesh.py` | ✅ | — | Run after caches from 02+03 |
| 5 | `05_affordance_head_debug.ipynb` | `models/` (stubs only) | ⬜ | **After step 4** | MLP + concat fusion; can use dummy features for API sketch only |
| 6 | `06_training_evaluation_debug.ipynb` | `training/`, `datasets/agd20k_*` | ⬜ | **No** | **AGD20K** annotations + end-to-end pipeline |
| — | `07_ablation_analysis.ipynb` | — | ⬜ | **No** | Phase 2+; MVP must pass first |

### Recommended next tasks (no SAM3D checkpoints)

| Priority | Task | Delivers |
|----------|------|----------|
| 1 | `05_affordance_head_debug.ipynb` | MLP + concat fusion on `vertex_semantic.pt` |

**Already landed (foundation):** repo scaffold (`src/`, configs), SAM3D wrapper (not runnable without weights), mesh I/O, `scripts/generate_sam3d.py`, `tests/test_mesh_loading.py`.

Later (Phase 2+): `07_ablation_analysis.ipynb` for mesh vs splat rendering and fusion ablations.

**Gate rule:** do not mark a step complete in the timeline until its notebook runs top-to-bottom on the target machine (mesh-only or full SAM3D) and the pass checklist is satisfied.

---

# Recommended MVP Timeline

---

# Step 0 — Bootstrap without SAM3D (current)

## Goal

Run rendering → VLM → projection using **only** a mesh asset.

## Input

* `data/sample.glb` (no matching splat required)

## Skip

* SAM3D inference until checkpoints are installed

## Verify

* mesh loads and normalizes
* mesh renderer produces sensible RGB / depth
* vertex correspondences are non-empty for visible vertices

## Notebook

`notebooks/00_mesh_bootstrap.ipynb` — load `data/sample.glb`, plot mesh, confirm normalization; optionally smoke-test mesh renderer once implemented.

---

# Step 1 — Reconstruction (when checkpoints available)

## Goal

Generate meshes (and optionally splats) from RGB-D inputs.

## Output

* `mesh.glb` (required)
* `gaussian.ply` (optional)
* vertex / SLAT latents
* camera parameters from SAM3D pose

## Verify

* reconstruction quality
* coordinate consistency between mesh and splat (if both exist)

## Notebook

`notebooks/01_reconstruction_debug.ipynb` — single-image SAM3D run; side-by-side mesh/splat preview; latent shapes documented.

---

# Step 2 — Novel View Rendering

## Goal

Render multiple views from mesh and/or splat.

## Output

For each view (per active backend):

* RGB
* depth
* vertex correspondences (from mesh rasterization; splat path may use mesh depth for projection)

## Modes

| Mode | When |
|------|------|
| `mesh` | No splat file — default for `sample.glb` |
| `gaussian` | `gaussian.ply` present |
| `both` | Ablation: same poses, two RGB sources |

## Verify

* view coverage
* visibility correctness
* mesh-only path works without any `.ply`

## Notebook

`notebooks/02_rendering_debug.ipynb` — grid of views; depth overlays; correspondence heatmap; compare `backend: mesh` vs `gaussian` when `.ply` exists.

---

# Step 3 — VLM Feature Extraction

## Goal

Extract patch embeddings.

## Output

```python
patch_features: [N_patches, D]
```

## Verify

* spatial alignment
* semantic consistency

## Notebook

`notebooks/03_vlm_features_debug.ipynb` — patch map dimensions; example patches on RGB; verb vector norm/similarity smoke test.

---

# Step 4 — 2D-to-3D Projection

## Goal

Assign patch features to vertices.

## Output

```python
vertex_vlm_features: [N_vertices, D]
```

## Verify visually

Color mesh by:

* PCA of features
* cosine similarity to verb

This debugging stage is extremely important.

## Notebook

`notebooks/04_projection_debug.ipynb` — **required** visual gate: PCA-colored mesh, verb-conditioned similarity map, before any training.

---

# Step 5 — Basic Affordance Head

## Goal

Train:

```python
MLP(vertex_feature) -> affordance
```

## Verify

* overfit tiny subset first
* inspect heatmaps

## Notebook

`notebooks/05_affordance_head_debug.ipynb` — train on ≤10 samples; loss curve; predicted affordance on mesh vs GT.

---

# Step 6 — Full MVP Evaluation

## Goal

Run on AGD20K.

## Metrics

* vertex accuracy
* IoU
* AUC

## Important

Only after this stage should you consider architectural complexity.

## Notebook

`notebooks/06_training_evaluation_debug.ipynb` — val metrics table; failure cases gallery; export paths for report figures.

---

# Phase 2 — First Real Improvements

Once MVP works:

---

## Extension A — Better VLM Features

### Add:

* intermediate transformer layers

### Motivation:

Better spatial precision.

---

## Extension B — Attention-Based Multi-View Fusion

Replace:

```python
mean(features)
```

with:

```python
attention(features)
```

### Motivation:

Learn view importance dynamically.

---

## Extension C — PointNet Global Features

Add:

* global object latent

### Motivation:

Improve holistic affordance reasoning.

---

## Extension D — Visibility-Aware Fusion

Add:

* confidence weighting
* occlusion filtering

### Motivation:

Reduce hallucinated affordances.

---

# Phase 3 — Advanced Geometric Reasoning

Only after previous stages work.

---

## Extension E — Transformer Vertex Head

Replace MLP with:

* transformer over vertices.

### Motivation:

Context-aware affordance reasoning.

---

## Extension F — Graph Neural Networks

Alternative to transformers.

Potentially more geometry-aware.

---

## Extension G — Gaussian Splat Semantic Fusion

Beyond using splats as an RGB render backend (MVP), use splats directly for:

* semantic accumulation on splat primitives
* differentiable rendering end-to-end.

---

# Phase 4 — Research-Level Contributions

Only after the pipeline is stable.

---

## Advanced Ideas

### 1. Open-vocabulary generalization

* unseen verbs
* compositional affordances

---

### 2. Uncertainty-aware affordances

Predict:

```python
(mean, variance)
```

---

### 3. Language-guided view selection

Actively render informative views.

---

### 4. Joint semantic-geometric training

Finetune reconstruction jointly.

---

# Recommended Actual Development Order

This is likely the safest order:

```text
0. Mesh-only bootstrap (sample.glb) + notebooks/00_mesh_bootstrap.ipynb
1. Mesh reconstruction (+ optional gaussian.ply) + notebooks/01_reconstruction_debug.ipynb
2. Multi-view rendering + notebooks/02_rendering_debug.ipynb
3. Frozen VLM extraction + notebooks/03_vlm_features_debug.ipynb
4. Patch-to-vertex projection + notebooks/04_projection_debug.ipynb
5. MLP affordance prediction + notebooks/05_affordance_head_debug.ipynb
6. AGD20K training + notebooks/06_training_evaluation_debug.ipynb
7. Report figures / interactive demos (reuse notebook outputs)
8. Intermediate VLM features
9. Attention fusion
10. PointNet global latent
11. Visibility-aware fusion
12. Transformer/GNN head
13. Splat-native semantic fusion (Extension G)
14. Open-vocabulary evaluation
```

This progression minimizes:

* debugging complexity,
* GPU cost,
* and architectural uncertainty,

while still preserving a clean path toward a publishable final system.
