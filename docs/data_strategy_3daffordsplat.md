# Data strategy: 3DAffordSplat + synthetic views + SAM3D

This document supersedes the earlier assumption that supervision would come from a dataset pairing **natural 2D photographs** with **dense 3D affordance ground truth** (e.g. AGD20K-style setups). Such paired data are scarce; instead we anchor on **3DAffordSplat** and a **synthetic imaging** loop.

## Paper and resources

**3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians** — Zeming Wei, Junyi Lin, Yang Liu, Weixing Chen, Jingzhou Luo, Guanbin Li, Liang Lin. arXiv:2504.11218 (April 2025).

* [arXiv abstract / PDF](https://arxiv.org/abs/2504.11218)
* Code: [HCPLab-SYSU/3DAffordSplat](https://github.com/HCPLab-SYSU/3DAffordSplat)
* Dataset distribution: [Hugging Face — Weizm/AffordSplat](https://huggingface.co/datasets/Weizm/AffordSplat) (dataset card mirrors the paper’s **3DAffordSplat** naming)

**What the dataset provides (high level, per the paper):**

* Large-scale **3D Gaussian Splatting (3DGS)** instances together with **point clouds**, aimed at affordance reasoning (continuous surfaces vs sparse points).
* **Manual affordance annotations** (paper: thousands of labels across **21** object categories and **18** affordance / interaction types), plus **language-guided Q&A** templates paired with object–affordance combinations.
* **Seen / Unseen** splits for generalization experiments.

The original AffordSplat / 3DAffordSplat line of work uses 3DGS as the *native* 3D representation for affordance learning. **This repo’s pivot** is different: we still want **SAM3D** (mesh + SLAT latents) and the existing **mesh-centric VLM → projection → MLP** stack, so we use 3DAffordSplat mainly as a **source of 3DGS (and labels)** and synthesize **multi-view 2D** (RGB, depth, masks) from the splat, then treat SAM3D as a **reconstruction module** from those views.

## Rationale

1. **2D + dense 3D affordance GT** for arbitrary real images is hard to obtain at scale.
2. **3DAffordSplat** offers **3DGS + affordance supervision + language**, which is enough to build a controlled pipeline: we know the underlying 3D representation and where affordances live on it.
3. **Rendering** from the official Gaussian splat gives **multi-view RGB** (and depth) with known **cameras** — a clean substitute for “captured” photos while we develop SAM3D integration and downstream heads.

## Target pipeline (conceptual)

```text
3DAffordSplat sample (3DGS + affordance labels [+ point cloud / language])
        ↓
Multi-view rendering from 3DGS (known intrinsics / extrinsics)
        ↓
(Optional) foreground / object masks (full object or tight crop; affordance masks from dataset can inform prompts or evaluation regions)
        ↓
SAM3D reconstruction from each (or a chosen) RGB (+ mask) bundle → mesh.glb + cached SLAT global latent
        ↓
Same as rest of repo: novel-view rendering (mesh and/or splat), VLM features, projection, verb-conditioned MLP
        ↓
Training / eval using 3DAffordSplat affordance labels, reprojected or aggregated onto **SAM3D mesh vertices** (requires a clear label transfer rule; see below)
```

## Design choices to lock down next

These are engineering decisions, not fixed by the paper:

| Topic | Options / notes |
|--------|------------------|
| **Which GS per sample** | Use the dataset’s native 3DGS asset; ensure export format matches our **Gaussian / mesh** render path (`.ply` or tooling from 3DAffordSplat). |
| **How many views** | Same ballpark as MVP (e.g. 4–8); can match spherical sampling used elsewhere for fairness. |
| **SAM3D input** | Paper-style single-image SAM3D: pick a **reference view** or run per-view and fuse (expensive); simplest MVP = **one reference RGB + mask** per object. |
| **Masks** | Full-object mask from alpha / segmentation vs affordance-region mask — affects what SAM3D reconstructs. |
| **Supervision transfer** | Affordances are defined on **3DAffordSplat’s** geometry (GS / PC). After SAM3D, labels must map to **mesh vertices** (nearest-point on surface, barycentric from ray-hit on GS, or render reprojection consistency). Document the chosen rule in the training notebook. |
| **Evaluation** | Optionally compare SAM3D mesh to **reference** geometry (GS-derived mesh or point cloud) as a **reconstruction** diagnostic separate from affordance metrics. |

## Implementation mapping (repo)

Planned additions (names may evolve):

* `src/datasets/` — loaders for on-disk **AffordSplat** / manifest data. **Implemented:** [`DataRootDataset`](../src/datasets/data_root_dataset.py) (JSONL manifest under `paths.data_root`) and [`AffordSplatLocalDataset`](../src/datasets/affordsplat_local_dataset.py) for a local HF mirror (`AFFORDANCE_AFFORDSPLAT_ROOT`, e.g. `/data`) — see [data_layout.md](data_layout.md).
* `scripts/` — e.g. batch **render views from GS** (`scripts/render_gaussian_views.py`), batch **invoke SAM3D** into `data/cache/sam3d/…`, export a **training manifest** (image paths, cameras, mesh path, verb, vertex label path).
* `configs/` — `configs/affordsplat.yaml` (or similar) pointing at `data/` roots and split JSONs from the upstream repo (`obj_aff_structure.json`, Seen/Unseen JSONs per 3DAffordSplat docs).
* `notebooks/07_training_evaluation_debug.ipynb` — end-to-end train/eval on this manifest instead of AGD20K.

## References

```bibtex
@misc{wei20253daffordsplat,
  title={3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians},
  author={Wei, Zeming and Lin, Junyi and Liu, Yang and Chen, Weixing and Luo, Jingzhou and Li, Guanbin and Lin, Liang},
  year={2025},
  eprint={2504.11218},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2504.11218}
}
```

Related follow-up work (scene-level sequential affordance on 3DGS) — **SeqAffordSplat** — is a different benchmark; we do not depend on it for the current MVP pivot.
