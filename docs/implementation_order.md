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
* no splat reasoning,
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
SAM3D reconstruction
        ↓
Mesh generation
        ↓
Render 4–8 novel views
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

* mesh output only

### Ignore for now:

* gaussian splats
* PointNet global features

### Goal:

Just get stable vertices and correspondences.

---

## 2. Rendering

### Keep:

* fixed spherical camera sampling
* 4–8 views

### Ignore:

* visibility confidence
* adaptive view selection

### Goal:

Generate enough coverage for semantic projection.

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
```

You do not yet need:

* experiments/
* graph models/
* transformer heads/
* uncertainty/
* advanced evaluation.

---

# Recommended MVP Timeline

---

# Step 1 — Reconstruction Only

## Goal

Generate meshes from RGB-D inputs.

## Output

* mesh.obj
* vertex features
* camera parameters

## Verify

* reconstruction quality
* coordinate consistency

---

# Step 2 — Novel View Rendering

## Goal

Render multiple views.

## Output

For each view:

* RGB
* depth
* vertex correspondences

## Verify

* view coverage
* visibility correctness

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

Use splats directly for:

* semantic accumulation
* differentiable rendering.

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
1. Mesh reconstruction
2. Multi-view rendering
3. Frozen VLM extraction
4. Patch-to-vertex projection
5. MLP affordance prediction
6. AGD20K training
7. Visualization/debugging
8. Intermediate VLM features
9. Attention fusion
10. PointNet global latent
11. Visibility-aware fusion
12. Transformer/GNN head
13. Gaussian splat integration
14. Open-vocabulary evaluation
```

This progression minimizes:

* debugging complexity,
* GPU cost,
* and architectural uncertainty,

while still preserving a clean path toward a publishable final system.
