This project investigates open-vocabulary affordance prediction by combining 3D reconstruction from SAM3D with semantic context extracted from a Vision-Language Model (VLM). The goal is to predict fine-grained, verb-conditioned affordance scores on a reconstructed 3D representation, enabling the model to reason about interactions such as *grasp*, *sit on*, *pour from*, or *open* in an open-vocabulary setting. The method combines geometric understanding from SAM3D with semantic reasoning from a frozen VLM to produce dense affordance fields over reconstructed objects.

**Data source + supervision (current plan):** dense 3D affordance ground truth paired with *natural* 2D images is hard to obtain at scale. Instead of hand labels we **distill a frozen teacher**: clean single-object images from **[OmniObject3D](https://omniobject3d.github.io/)** (Wu et al., 2023) are reconstructed with **SAM3D**, and **[GEAL](https://github.com/DylanOrange/geal)** (Lu et al., CVPR 2025) generates per-vertex affordance **pseudolabels** on that reconstructed geometry. Because GEAL labels the *same* mesh we train on, **no cross-modal alignment is required** — this replaces the earlier 3DAffordSplat + ICP-alignment plan, which failed because SAM3D reconstructs splat renders poorly. Full rationale, links, and the taxonomy-overlap constraint are in [data_strategy_geal_omniobject3d.md](data_strategy_geal_omniobject3d.md).

The proposed pipeline consists of the following stages:

---

## 1. Input Processing

The system receives as input a single RGB-D image together with an open-vocabulary verb describing a potential interaction. The RGB-D image provides both appearance and geometric cues, while the verb acts as a semantic query conditioning the affordance prediction process.

In the **OmniObject3D-driven** setup, that observation is a clean single-object image (a real scan render). SAM3D reconstructs the mesh used downstream; supervision comes from **GEAL pseudolabels** computed on a point cloud sampled from that mesh and mapped to its vertices (see [data_strategy_geal_omniobject3d.md](data_strategy_geal_omniobject3d.md)).

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

In addition to geometry, SAM3D's structured latent (SLAT) diffusion stage produces a sparse set of features `(N, 8)` over the N occupied voxels of the reconstructed object, where each voxel carries both geometric and appearance information. These are mean-pooled over occupied voxels to produce a single **global object latent** of shape `(8,)`. This vector captures holistic structure — shape, rough appearance, and object-scale layout — and is broadcast identically to every mesh vertex during feature fusion.

SLAT is preferred over the earlier SS (sparse structure) latent because: (1) it only covers occupied voxels, avoiding dilution from empty space; and (2) it captures geometry-and-appearance jointly, which is more discriminative for affordance prediction than pure occupancy structure alone. The VLM patch features and SLAT are complementary: VLM sees 2D rendered views from fixed viewpoints; SLAT was trained on the full 3D structure.

**Latent caching:** Because SAM3D inference is expensive, the `(8,)` global latent is computed once per object and saved to `data/cache/sam3d/<object_stem>/global_latent.pt`. The training loop loads from cache and never re-runs SAM3D. Cache files are written by `reconstruction.sam3d_wrapper.save_global_latent` and read by `load_global_latent`.

---

## 3. Novel View Sampling and Rendering

To obtain richer semantic coverage than available from the original observation, multiple novel viewpoints are sampled around the reconstructed object.

### Rendering modes

Novel views for the VLM use a **hybrid** path by default (`rendering.backend: gaussian` or `both`):

| Piece | Source |
|--------|--------|
| Depth, `vertex_uv`, `vertex_visible` | Mesh rasterization (pyrender) |
| RGB | 3DGS `.ply` via **gsplat** when CUDA + `gsplat` are available; otherwise pyrender **centre preview** |
| `normal_rgb`, `depth_vis_rgb` (optional) | Extra **mesh** passes: world normals as RGB + depth as 3× grayscale (`render_geometry_aux`, default **true**) — sharp edges for SAM when splat RGB is muddy |

Orbit pitch is enforced in **`[25°, 60°]`** above the ground plane (intersection with YAML bounds): no near-horizon cameras, moderate downward views toward the asset center.

With `backend: mesh`, RGB also comes from the mesh. If `splat_path` is missing or splat rendering fails, `gaussian` / `both` **fall back** to mesh RGB (warning). Projection always targets **mesh vertices**.

For local work with only `data/sample.glb`, set `backend: mesh` or `splat_path: null` if you want mesh-colour RGB without pairing a splat file.

Each rendered view provides:

* RGB image (for the frozen VLM or SAM-style models),
* linear **depth** map (float32, mesh-aligned in hybrid mode),
* optional **`normal_rgb`** / **`depth_vis_rgb`** (uint8 RGB) from the mesh for boundary cues,
* visibility / foreground mask,
* camera intrinsics and extrinsics,
* **vertex correspondences** (mesh UV/barycentric or projected vertex indices) for 2D→3D projection.

Initially, viewpoint sampling uses uniform azimuth on a fixed-elevation ring around the object, with elevation restricted to **[25°, 60°]** (see `rendering/camera_sampling.py`). Later iterations may explore visibility-aware or coverage-optimized view selection.

**MVP note:** vertex affordances remain mesh-based; splats supply **RGB** for VLMs when configured, not per-Gaussian labels.

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

For every mesh vertex, three information sources are concatenated into a joint feature vector:

```
vertex_feature = concat(
    vlm_feature,      # (512,) — per-vertex, from 2D→3D projection
    verb_embedding,   # (512,) — global, broadcast to all vertices
    sam3d_global,     # (8,)   — global, broadcast to all vertices
)                     # total: (1032,)
```

* **`vlm_feature`** carries local visual-semantic information tied to the vertex's surface appearance across rendered views.
* **`verb_embedding`** encodes the interaction intent (e.g. *grasp*, *pour from*).
* **`sam3d_global`** encodes the object's overall geometric structure, shared across all vertices.

The two broadcast terms (`verb_embedding`, `sam3d_global`) give every vertex access to global context; the `vlm_feature` distinguishes vertices from one another.

**MVP note:** when SAM3D checkpoints are unavailable, `sam3d_dim` is set to `0` in config and `sam3d_global` is omitted, reducing input dim to `1024`. The MLP architecture handles this transparently via `MLPHeadConfig.sam3d_dim`.

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

The system is trained using per-vertex affordance **pseudolabels** from the frozen **GEAL** teacher, computed on a point cloud sampled from each SAM3D mesh and mapped to its vertices — already in mesh-vertex order, so **no label alignment step is needed** (see [data_strategy_geal_omniobject3d.md](data_strategy_geal_omniobject3d.md)).

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
