This project investigates open-vocabulary affordance prediction by combining 3D reconstruction from SAM3D with semantic context extracted from a Vision-Language Model (VLM). The goal is to predict fine-grained, verb-conditioned affordance scores on a reconstructed 3D representation, enabling the model to reason about interactions such as *grasp*, *sit on*, *pour from*, or *open* in an open-vocabulary setting. The method combines geometric understanding from SAM3D with semantic reasoning from a frozen VLM to produce dense affordance fields over reconstructed objects.

The proposed pipeline consists of the following stages:

---

## 1. Input Processing

The system receives as input a single RGB-D image together with an open-vocabulary verb describing a potential interaction. The RGB-D image provides both appearance and geometric cues, while the verb acts as a semantic query conditioning the affordance prediction process.

The goal is to infer a probabilistic affordance value in the range ([0,1]) for every vertex of the reconstructed object, representing the likelihood that the queried interaction can be performed at that location.

---

## 2. 3D Reconstruction using SAM3D

The RGB-D image is processed using SAM3D to generate a dense 3D representation of the scene or object. The reconstruction can produce:

* a textured **mesh** (`.glb` / `.obj`) — required; this is the canonical geometry for affordance prediction and projection,
* an optional **Gaussian splat** (`.ply`) — used when available for higher-fidelity novel-view rendering.

**Affordance prediction always lives on mesh vertices.** Splats are a rendering backend for VLM input images, not a separate affordance representation.

### Development without SAM3D checkpoints

Until SAM3D weights are available, the pipeline can be bootstrapped with a pre-exported mesh only (e.g. `data/sample.glb`). In that mode:

* skip reconstruction,
* use **mesh rendering** for novel views,
* omit splat files entirely — no paired `.ply` is required.

When checkpoints are available, SAM3D outputs both mesh and splat; rendering can then use either or both backends (see §3).

In addition to geometry, SAM3D produces latent geometric features associated with mesh vertices or reconstructed points. To obtain stronger global geometric context, these local features can optionally be aggregated using a PointNet-style encoder that produces:

* local geometric features per vertex,
* and a global object-level latent representation.

The global latent is intended to capture object-scale semantic and structural information that may be important for affordances requiring holistic reasoning, such as *sit on* or *pour from*.

---

## 3. Novel View Sampling and Rendering

To obtain richer semantic coverage than available from the original observation, multiple novel viewpoints are sampled around the reconstructed object.

### Dual rendering backends

Novel views for the VLM are produced by one or both of:

| Backend | Input | Typical use |
|--------|--------|-------------|
| **Mesh renderer** | `.glb` / `.obj` | Always available; provides depth + per-vertex 2D correspondences via rasterization |
| **Gaussian splat renderer** | `.ply` from SAM3D | Optional; often sharper RGB when a splat exists |

Configuration selects the active backend(s), e.g. `mesh`, `gaussian`, or `both`. When `both` is set, the same camera poses can render two RGB streams (mesh vs splat) for comparison or ablation; projection still targets **mesh vertices**.

If only a mesh is present (no splat file), the pipeline runs in **mesh-only** mode — this is the expected setup for local development with `data/sample.glb`.

Each rendered view provides:

* RGB image (for the frozen VLM),
* depth map,
* visibility / foreground mask,
* camera intrinsics and extrinsics,
* **vertex correspondences** (mesh UV/barycentric or projected vertex indices) for 2D→3D projection.

Initially, viewpoint sampling uses simple uniform spherical sampling around the object. Later iterations may explore visibility-aware or coverage-optimized view selection.

**MVP note:** mesh rendering is sufficient to validate projection and affordance heads; splat rendering is added when `gaussian.ply` exists alongside `mesh.glb`.

---

## 4. Vision-Language Feature Extraction

Each rendered view is processed by a frozen Vision-Language Model together with the queried verb.

The VLM extracts:

* patch-level visual embeddings from the rendered images,
* and a text embedding representing the semantic meaning of the verb.

The initial implementation uses feature concatenation for conditioning, where the verb embedding is concatenated with projected visual and geometric features. This provides a lightweight and stable baseline for open-vocabulary affordance prediction.

Initially, patch embeddings from the final VLM layers can be used directly. However, since deeper layers may lose spatial precision, later experiments may investigate extracting features from intermediate transformer layers to better preserve:

* local geometry,
* contact regions,
* handles,
* openings,
* and interaction-relevant structures.

Future extensions may additionally explore:

* cross-attention between verb embeddings and visual features,
* multi-scale feature extraction,
* or semantic attention maps from the VLM.

---

## 5. Projection of VLM Features onto the 3D Representation

Using the rendering correspondences between image space and 3D geometry, the VLM patch embeddings are projected back onto the reconstructed 3D object.

Projection is performed by mapping visible 3D points or mesh vertices into image space and associating them with corresponding VLM patch features. Features from multiple rendered views are aggregated per vertex to produce a unified semantic representation over the mesh.

The initial implementation may use simple aggregation strategies such as:

* averaging,
* max pooling,
* or weighted averaging.

Subsequent iterations will investigate attention-based multi-view feature fusion, where the model learns to combine semantic evidence from multiple viewpoints dynamically.

Visibility information obtained during rendering is used to ensure that only geometrically visible vertices receive semantic updates from a given view. More advanced visibility-aware confidence weighting and uncertainty estimation may later be explored to reduce hallucinated affordance predictions in occluded regions.

---

## 6. Vertex Feature Fusion

For every mesh vertex, multiple information sources are fused into a joint feature representation:

* SAM3D geometric latent,
* projected VLM semantic latent,
* optional global PointNet latent,
* and the verb embedding.

This creates a unified geometric-semantic representation capturing:

* local shape information,
* object-scale context,
* visual semantics,
* and interaction intent.

The fused representation forms the basis for downstream affordance prediction.

---

## 7. Affordance Prediction Head

The initial affordance predictor is implemented as a lightweight multilayer perceptron (MLP) operating independently on each vertex feature.

The MLP predicts a probabilistic affordance score in the range ([0,1]) for every mesh vertex.

This lightweight formulation serves as a strong baseline while keeping the overall system computationally efficient and modular.

Future iterations may replace the MLP with more expressive architectures, including:

* transformer-based vertex reasoning,
* graph neural networks over mesh connectivity,
* or geometry-aware message passing networks.

These extensions may improve spatial consistency and enable more sophisticated interaction reasoning across neighboring surface regions.

---

## 8. Training and Supervision

The system is intended to be trained using affordance annotations from datasets such as AGD20K.

Training supervision is formulated as probabilistic vertex-wise affordance prediction conditioned on the queried verb.

The initial training setup will likely use binary cross-entropy loss on per-vertex affordance labels. Later extensions may additionally incorporate:

* spatial smoothness regularization,
* visibility-aware consistency losses,
* multi-view consistency objectives,
* or contrastive language-geometry alignment losses.

---

## 9. Output Representation

The final output is a dense, verb-conditioned affordance field defined over the reconstructed 3D mesh.

Each vertex receives a continuous affordance score representing the predicted likelihood that the queried interaction can be performed at that location. High-scoring regions correspond to semantically and geometrically plausible interaction areas, enabling fine-grained 3D reasoning about object affordances in an open-vocabulary setting.

---

## 10. Stage testing notebooks

The pipeline is developed and validated **stage by stage** using Jupyter notebooks under `notebooks/`. Each major step (mesh bootstrap, reconstruction, rendering, VLM, projection, affordance head, evaluation) has a dedicated notebook that:

* runs that stage in isolation using `configs/default.yaml`,
* visualizes intermediate results (meshes, render grids, feature-colored surfaces, affordance heatmaps),
* and documents pass/fail criteria before the next stage starts.

Unit tests in `tests/` complement notebooks but do not replace visual inspection of geometry and semantics. See [implementation_order.md](implementation_order.md) for the full notebook ↔ stage mapping and gate rule.
