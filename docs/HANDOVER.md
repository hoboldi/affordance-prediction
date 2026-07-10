# Project Handover — Verb-Conditioned 3D Affordance Prediction

_Last updated: 2026-07-10. Read this top-to-bottom before touching anything._

---

## 0. TL;DR

**Goal.** From a **single 2D image + a verb**, predict a **per-vertex 3D affordance map** on a mesh
reconstructed by SAM3D. A verb-conditioned spatial GNN head is **distilled from a frozen GEAL teacher**
(pseudolabels) then **finetuned on ~200 human-labeled objects**. Metric = **macro (verb-balanced)
AUPRC**, 5-fold CV.

**Current phase.** Research/experiments are **essentially complete**. Active work = writing the
**4-page paper** and finalizing the **poster**. No critical experiment remains except finishing one
ablation cell (`noboth`, §7).

**The headline finding (the paper's thesis):** affordance verbs split into **two classes** —
*geometric* (shape-driven) vs *appearance* (learned-visual-driven) — which differ in which feature
they need and which generalize across object categories.

### ⚠️ READ FIRST — ephemeral code
The **training harness and CV splits are NOT in the repo.** They live in a `/tmp` session scratchpad
that will be deleted:

```
/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad/
```

Contains (all EPHEMERAL — persist before relying on them):
`train_gnn_cv.py` (main harness), `train_gnn_pretrain.py`, `cv_fold{0-4}.json` (the CV splits),
`manifest.cv.jsonl` (training manifest), and every analysis script (`geal_scan_8v.py`,
`synonym_test_*.py`, `crossverb_iou_8v.py`, `eval_pretrained_head.py`, `run_*.sh`, …).

**First action for the next agent: copy these into the repo** (e.g. `scripts/cv/` + `docs/splits/`).
Checkpoints (`outputs/*.pt`), human labels, reconstructions, and precompute scripts (`scripts/`) **are**
persistent — only the harness + splits + eval scripts are at risk.

---

## 1. The Model (all dims confirmed from code)

**Per-vertex input = 261-d** = DINO 128 + geometry 5 + verb-text 128.

| channel | dim | source | file |
|---|--:|---|---|
| **DINO appearance** | 128 | DINOv2-base (`facebook/dinov2-base`), render mesh multi-view @448px→32×32 patches, back-project **facing-weighted mean** across visible views (`clip(n·d_cam, 0.1, 1)`), ℓ₂-norm, **random Gaussian projection 768→128** (not PCA/learned) | `scripts/generate_vertex_dino.py` → `vertex_dino_fine.pt` |
| **Geometry** | 5 | height_up, concavity `mean(x_j−x_i)·n_i`, curvature `mean(1−n_i·n_j)`, normal_up `n·ŷ`, radial extremity. ch1–4 per-object standardized; ch0 ∈[0,1]. Computed on canonical (gravity-aligned) frame, 8-NN | `scripts/precompute_geom.py` → `vertex_geom.pt` |
| **Verb** | 128 | CLIP **text** encoder (512-d) → linear proj 128. *CLIP-vision is NOT used (dropped).* | in-model |

**Dropped by ablation (≈0 AUPRC):** CLIP-vision (128), SLAT (8), surface normals (3).

**Head** (`src/models/gnn_head.py`, `AffordanceGNN`): EdgeConv, kNN graph **k=24**, **3 layers**,
hidden **128**, **max** aggregation, `verb_in_backbone=True` (verb injected at every layer), sigmoid out.
Projection back-projection code: `src/projection/project_to_mesh.py`.

**Training (2 stages):**
1. **Distill** — frozen GEAL teacher pseudolabels → pretrain head (soft-BCE). Ckpt `gnn_pretrain_k24_8v.pt`.
2. **Finetune** — warm-start, human labels, **class-balanced BCE**, 5-fold CV over objects.

---

## 2. Numbers of Record (8 verbs, clean human labels, 5-fold macro AUPRC)

**Final model = `lean8` (DINO+geom+verb, 261-d) = 0.863.** Per-fold 0.872/0.868/0.864/0.820/0.851.

Per-verb: contain 0.98 · sit 0.92 · press 0.91 · display 0.91 · pour 0.89 · move 0.86 · grasp 0.74 · lift 0.70.

**Teacher comparison (macro):** GEAL 0.46 → Ours **0.86 (+0.40)**. Beats GEAL on every verb; press/lift
biggest wins. ⚠️ **Frame this honestly in the paper:** GEAL is evaluated **zero-shot on our data** — it
is a *distillation teacher / transfer upper-bound*, NOT a head-to-head SOTA competitor. Do not write
"we outperform GEAL."

**Component ablation (Δ from lean8; All / Geometric / Appearance):**
| removed | All | Geo | App |
|---|--:|--:|--:|
| verb conditioning | −0.39 | −0.40 | −0.38 |
| **DINO** | −0.13 | −0.06 | **−0.25** |
| **geometry** | −0.07 | **−0.09** | −0.04 |
| spatial GNN (→MLP) | −0.06 | −0.05 | −0.09 |
| DINO+geom (both) | → ~0.18 | ~0.18 | ~0.18 **[EXTRAPOLATED — see §7]** |

Absolute: lean8 0.863 · −geom 0.793 · −GNN 0.803 · −DINO 0.735 · −verb 0.472 · −both ~0.18 (chance ≈0.20).

**GNN vs MLP:** 0.814 → **0.870** (+0.056). **Cross-category (LOCO):** pour 0.82 ✓ / grasp 0.15 ✗.
**Cross-verb region IoU:** max 0.059 (verbs occupy near-disjoint regions → use as *verb-necessity*
support in the ablation, NOT as a standalone "generalization" claim).

---

## 3. THE FINDING — two classes of affordance verb

- **Geometric** (shape-determined): contain, pour, sit, move, **lift**. Rely on geometry (−0.09), not DINO (−0.06). Transfer across categories.
- **Appearance** (learned-visual): grasp, display, **press**. Rely on DINO (−0.25), not geometry (−0.04). Don't transfer.
- ⭐ **lift vs grasp** = the star illustration: both "handle" verbs, opposite classes (lift=geometric strap, grasp=appearance). Same action, different object → different class.
- Mechanistic footnote: the 5 hand-crafted geometry features literally align with geometric verbs
  (concavity↔contain/pour, normal_up↔sit, radial↔grasp/lift extremities); appearance verbs must
  recover their cue from learned DINO latents.
- **Caveat to keep:** borderline verbs are partly object-dependent; small-n verbs (lift n=12) are noisy →
  report mean±std, frame as *evidence for* not *proof of* the dichotomy.

---

## 4. Language / open-vocab investigation — CLOSED (negative, but useful)

Tested three ways, all agree: **the model is NOT open-vocabulary.**
- **Exact verbs + trained synonyms:** robust (SEEN synonyms recover to −0.01 after augmentation).
- **Held-out (novel) synonyms:** baseline syn-aug −0.30; richer augmentation (6 syn × 4 templates,
  5-fold) only 0.555 → **0.591 (+0.036)** — helps 2 common words (rest on, keep inside), leaves
  rare/polysemous words dead (decant 0.10, click 0.33, elevate 0.18).
- **Test-time prompt-ensembling:** null / slightly worse (−0.07).
- **Bottleneck = CLIP text geometry** (polysemy), not our head. **Not worth more compute.**
- **Poster:** one clause in Limitations. **Paper:** a proper Analysis/Limitations subsection.
- Ckpts: `gnn_synaug8_fold*.pt` (4-paraphrase), `gnn_synaugrich8_fold*.pt` (rich). These are
  **experiments, not the final model** — the shipped model is `lean8` (no augmentation).

---

## 5. Data & Labels

- **Reconstructions + features:** `/home/datasets/customDatasets/cmr2/reconstructions/{obj}/` —
  `mesh.glb`, `vertex_positions`, `vertex_normals.pt`, `vertex_dino_fine.pt`, `vertex_geom.pt`,
  `vertex_knn_k{8,24}.pt`, `vertex_pseudolabels_{verb}.pt` (GEAL teacher), `canonical_rotation.pt`.
- **Human labels:** `human_gt_labels/{obj}/vertex_manuallabels_{verb}.pt`. **NEVER overwrite GEAL
  pseudolabels with these.**
- **Clean-eval filter:** some human labels were contaminated with GEAL pseudolabels (continuous, not
  binary). Filter with `frac_intermediate = ((h>0.001)&(h<0.999)).mean() < 0.005` — used in ALL evals.
- **Scale:** ~200–238 CO3D objects, 9 categories, **8 verbs**
  (contain, sit, pour, move, display, grasp, press, lift), 5-fold CV.
- HF dataset: `MattisKr/co3d-sam3d-affordance` (meshes-only now). glb vertex order == vertex_positions order.

---

## 6. Deliverables & where things live

| item | path | status |
|---|---|---|
| Poster script (numbers/captions, 8-verb) | `final_figures/POSTER.md` | current; layout sketched, GNN-vs-MLP added |
| Figure inventory | `final_figures/MANIFEST.md` | all figures + regen commands |
| All figures/renders | `final_figures/` | **always put new figures here, never `outputs/renders/`** |
| Consolidated results report (6-verb) | `docs/RESULTS_consolidated.md` | committed on `dev` |
| Poster layout wireframe | Artifact `a50dcd7f-…` (this session) | structural sketch |
| Paper outline + Method §3.2/§3.3 drafts | (in conversation — not yet filed) | **write to `docs/PAPER_outline.md`** |

**Checkpoints (`outputs/`):** `gnn_lean8_fold{0-4}.pt` (final), `gnn_pretrain_k24_8v.pt` (distilled),
`gnn_leanabl8_{nodino,nogeom,noverb,noboth}_fold*.pt` (ablations), `gnn_leanmlp8_fold*.pt` (MLP),
`gnn_synaug8_fold*.pt` / `gnn_synaugrich8_fold*.pt` (synonym experiments),
`gnn_full_preclean_fold{0-4}.pt` (0.895 full-feature model used for figure renders).

---

## 7. Open items / next steps (prioritized)

1. **PERSIST the ephemeral scratchpad** (§0) — do this first or the rest is unreproducible.
2. **Finish `noboth`** (DINO+geom ablation) — stalled at 3/5 folds; run folds 3,4 (~30 min) to replace
   the extrapolated ~0.18 cell with a measured one for the paper's ablation table.
3. **Paper (4 pages):** structure = Abstract+Intro / Related Work / Method / Experiments / Conclusion.
   Method §3.2 is confirmed & drafted (§1 above). TODO: related work, honest GEAL framing, **mean±std**
   on headline numbers, dataset/cleaning prose.
4. **Poster fixes (a reviewer will catch these):**
   - Pipeline "CLIP" box → **"CLIP-text (verb)"** (currently reads as the dropped CLIP-vision — contradiction).
   - Reconcile **"133-d" vs "261-d"** (133 = DINO+geom features; 261 = +verb). Pick one.
   - Say **"8 verbs"** everywhere (not "6–8").
   - Consistency sweep: grasp 0.78 vs 0.74, DINO-App −0.32 vs −0.25 across table/bars/two-class.
5. **Optional:** karahi in-the-wild demo (`final_figures/karahi_gt_vs_pred.png`, contain 0.66 / grasp 0.89)
   — closes the Motivation→Results real-world gap; user chose GNN-vs-MLP over it for the poster.

---

## 8. Operational constraints & gotchas (IMPORTANT)

- **Shared GPU box.** `nvidia-smi` FIRST. Other users' jobs run in parallel (do not preempt). Memory is
  usually free (49 GB cards) — sharing compute is OK, **just don't OOM/crash them**. Small GNN jobs
  fit in ~2 GB. Use free GPUs or CPU for eval (`CUDA_VISIBLE_DEVICES=""`).
- **Metric = MACRO (verb-balanced), not micro.** Micro is contain-dominated (~0.92 inflated). Always macro.
- **Git safety.** Shared checkout + live training share this tree. Commit **local on `dev` only — NO push,
  NO Claude co-author trailer.** Merge with `-s ours` or a worktree so training-critical files aren't
  rewritten. SSH key is passphrase-locked; no `gh`.
- **Figures → `final_figures/`** (curated), never `outputs/renders/`. Follow MANIFEST naming.
- **Human labels → `human_gt_labels/`**, never overwrite GEAL `vertex_pseudolabels_*`.
- **CUDA driver is flaky** — wrap training launches in a 2–3× retry (see `run_*.sh` pattern); use
  `setsid` to detach long runs; background evals block-buffer stdout through `grep|tail` (output appears
  only at completion — normal).
- **Affordance-map renders:** use FiLM ckpt (`geomclean/lc_*`, NOT `gnn_geom`); pass
  `dino_cls + ss_dino_cls + vertex_dino_fine.pt`; `vmax = max(.05, p99.5)`.

---

## 9. Reproduce a run (once scratchpad is persisted)

```bash
# Train one fold of the final model (warm-start from distilled pretrain)
CUDA_VISIBLE_DEVICES=<free> python <SCR>/train_gnn_cv.py \
  --drop clip,slat,normals --init_from outputs/gnn_pretrain_k24_8v.pt \
  --folds 0 --knn_k 24 --tag lean8 --device cuda

# Evaluate teacher vs ours (CPU): <SCR>/geal_scan_8v.py
# Ablations: add --drop dino / --drop geom / --no_verb / --gnn_layers 0 (MLP)
```
Env: `.venv-affordance/bin/python`, `PYTHONPATH=src`, working dir `/home/kraum/Prototype`.
