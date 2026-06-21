# Data strategy: GEAL pseudolabels on OmniObject3D

This supersedes the earlier **3DAffordSplat + ICP-alignment** plan. That approach rendered multi-view
RGB from the dataset's 3D Gaussian splats, reconstructed a mesh with SAM3D, then transferred affordance
labels from the splat/point-cloud frame onto the SAM3D mesh with a normalize → rigid-ICP →
nearest-neighbour step. It broke down because **SAM3D reconstructs splat renders poorly**, so the ICP
alignment landed labels on the wrong vertices.

## New approach

1. **Source data — OmniObject3D.** Real-scanned, clean, single-object captures (textured meshes, point
   clouds, multi-view Blender renders). Because objects are isolated and clean, **SAM3D reconstructs
   them well** — the property the pivot is built on.
2. **Teacher — GEAL.** A frozen [GEAL](https://github.com/DylanOrange/geal) (CVPR 2025) model generates
   affordance **pseudolabels**. Only its 3D branch (`model.branch_3d.Branch3D`) is used at inference: a
   PointNet++ + RoBERTa model that maps a normalized point cloud + an affordance question to per-point
   probabilities in `[0, 1]`. The Gaussian-splatting 2D branch is training-only, so **no CUDA rasterizer
   is needed** for labeling.
3. **No alignment.** GEAL labels the **same geometry we train on** (a point cloud sampled from the SAM3D
   mesh), then scores are mapped to mesh vertices by nearest sampled point — intra-mesh interpolation in
   one frame, not a cross-modal transform.

```text
OmniObject3D image
   → SAM3D reconstruct (mesh.glb + SLAT global latent)
   → sample ~2048 surface points from the mesh
   → GEAL(point cloud, affordance question) → per-point scores in [0,1]
   → nearest-point map to mesh vertices → vertex_pseudolabels.pt  (V,)
   → train: VLM patch features → vertices ⊕ verb ⊕ SAM3D latent → MLP → per-vertex affordance
```

## Taxonomy overlap (the binding constraint)

GEAL is trained on **PIAD** and **LASO**, which both inherit 3D-AffordanceNet's **23 object** and
**18 affordance** classes (see `labeling.GEAL_CLASSES` / `labeling.GEAL_AFFORDANCES`). GEAL produces
meaningful labels **only** for objects in this taxonomy, so OmniObject3D is filtered to overlapping
categories via `category_map` in [../configs/omniobject3d.yaml](../configs/omniobject3d.yaml). Each kept
class is queried with one canonical affordance (`affordance_map`) for the MVP; multiple affordances per
object is a later extension.

The published OmniObject3D docs do not list all 190 category names, so `category_map` is a best-effort
table keyed by folder name (lowercased) — **verify and extend it against the real download**.

## Repo mapping

| Step | Code |
|------|------|
| Stage images + manifest | `scripts/prepare_omniobject3d.py`, `src/datasets/omniobject3d.py` |
| SAM3D reconstruction | `scripts/generate_sam3d.py`, `src/reconstruction/sam3d_wrapper.py` |
| GEAL pseudolabels | `scripts/generate_geal_pseudolabels.py`, `src/labeling/geal_infer.py` |
| Training | `scripts/train_affordance.py`, `src/training/vertex_affordance_train.py`, `src/datasets/data_root_dataset.py` (`vertex_pseudolabel_path`) |

## OmniObject3D layout (per object)

```
<root>/blender_renders/<category>/<object_id>/render/{images,depths,normals,transforms.json}
<root>/raw_scans/<category>/<object_id>/Scan/Scan.obj
<root>/point_clouds/...   # PLY/HDF5 at 1024 / 4096 / 16384 points
```

The ingestion stages one (or more) render image per object into `images/` + `masks/` (mask from the
render alpha when present) for `generate_sam3d.py`.

## Evaluation note

Pseudolabel distillation has no perfect end-to-end real GT for the image→reconstruct→head pipeline.
PIAD/LASO ground truth can be used to **validate the teacher** (run GEAL on their point clouds and compare
to GT — reproduces the paper), which is the meaningful sanity check on label quality.

## References

- GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency — Lu, Kong, Huang, Lee, CVPR 2025. [arXiv:2412.09511](https://arxiv.org/abs/2412.09511) · [code](https://github.com/DylanOrange/geal) · [weights](https://huggingface.co/datasets/dylanorange/geal)
- OmniObject3D: Large-Vocabulary 3D Object Dataset — Wu et al., CVPR 2023. [arXiv:2301.07525](https://arxiv.org/abs/2301.07525) · [project](https://omniobject3d.github.io/)
- 3D-AffordanceNet (taxonomy) — Deng et al., CVPR 2021. [arXiv:2103.16397](https://arxiv.org/abs/2103.16397)
