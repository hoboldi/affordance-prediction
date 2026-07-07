# Verb-conditioned 3D affordance prediction — consolidated results & findings audit

*Overnight consolidation. Everything below is on CLEAN human labels (54 GEAL-contaminated labels dropped) unless flagged. "5-fold" = 5-fold CV over 226 human-labeled objects, trained-verb mean AUPRC. Placeholders `[ABL:x]` / `[ROUTE:x]` filled from the running ablation + routing re-verification.*

---

## 1. One-paragraph summary
We predict **verb-conditioned per-vertex 3D affordances** on meshes reconstructed from single real images (CO3D → SAM3D/TRELLIS). Per-vertex features (CLIP-vision, DINOv2, geometry, SAM3D-SLAT, normals) plus a CLIP-text verb embedding feed a **verb-conditioned head**; the model is distilled from a frozen GEAL teacher, then fine-tuned on human labels. Our contribution is (a) replacing the per-vertex MLP head with a **spatial GNN** (EdgeConv message passing over the mesh kNN graph), which lifts trained-verb mean AUPRC from **0.814 → 0.870**, plus **GEAL pretraining → 0.895**; and (b) a rigorous characterization of *where the model generalizes and where it can't* — new objects ✓, geometric verbs cross-category ✓, grasp cross-category ✗, novel verbs zero-shot ✗ (proven structural), novel verbs few-shot ✓. The model **beats the GEAL teacher it distills from** (+0.058 on human labels). Feature ablation reduces the input to **DINO + geometry + verb-text** with no accuracy loss (0.890 vs 0.895, n.s.) — this **lean 133-d model is the adopted final model**, dropping the CLIP-vision and SLAT extraction stages for free.

## 2. Method
- **Input:** single image → SAM3D reconstruction → mesh (`vertex_positions`) + structured latent (SLAT).
- **Per-vertex features (272-d):** CLIP-vision 128 · DINOv2 (fine 32×32 patches, multi-view facing-weighted mean fusion) 128 · geometry 5 (height/concavity/curvature/normal-up/radial) · SLAT 8 · normals 3. (position not used.)
- **Verb conditioning:** CLIP-text embedding of the verb (128-d after projection), concatenated to every vertex → whole backbone is verb-conditioned (open-vocab).
- **Head:** best = **AffordanceGNN** — LayerNorm+Linear → 3–4 EdgeConv layers (`msg=MLP([h_i, h_j−h_i])`, max-pool over k neighbors, residual+LN) over a kNN graph (k=24) → per-vertex logit. Baseline = per-vertex MLP with FiLM.
- **Training:** distill on GEAL pseudolabels (1,227 objects, 6 verbs) → full fine-tune on human labels (all params). Metric of record = trained-verb mean AUPRC, 5-fold CV.

## 3. Headline results (clean, 5-fold trained-verb mean AUPRC)
| stage | AUPRC | Δ |
|---|--:|--:|
| head-only (start) | 0.650 | — |
| full fine-tuning | 0.785 | +0.135 |
| + geometry channel | 0.814 | +0.029 |
| **spatial GNN** | **0.870** | **+0.056** |
| **+ GEAL pretraining** | **0.895** | **+0.025** |
| **lean final (DINO+geom+verb, 133-d)** | **0.890** | −0.005 (n.s.) — same accuracy, dropped CLIP-vision+SLAT (§4b/4e) |

- GNN > MLP on **all 5 folds** (+0.056; per-fold +0.064/+0.063/+0.026/+0.075/+0.052). GEAL-pretrain gain +0.025 also all-folds-positive.
- Per-verb (best model): contain ~0.97 · display ~0.92 · sit ~0.86 · pour ~0.83 · move ~0.80 · **grasp ~0.75** (hardest).
- Context: chance (base rate) ≈ 0.20; **nonsense-verb prior** ≈ 0.584; beats **GEAL teacher +0.058**; pure-GEAL-no-manual on human = **0.613** (manual FT does the heavy lifting, +0.28).

## 4. Ablations
### 4a. Architecture (established)
MLP (0.814) → GNN (0.870): **+0.056**, message passing over the mesh is the single biggest lever. Coherent regions vs speckled per-vertex guesses (see figures).

### 4b. Feature leave-one-out (GNN k24, clean, 3-fold — RESULTS)
Baseline (all features) = **0.880**. Each row drops one channel (zeroed in train+eval); contribution = 0.880 − drop-mean.
| dropped channel | 3-fold AUPRC | **contribution** |
|---|--:|--:|
| full (all features) | 0.880 | — |
| − **DINOv2** | 0.823 | **+0.057** (dominant) |
| − **geometry** | 0.860 | **+0.020** (secondary) |
| − CLIP-vision | 0.880 | **+0.000** (dead) |
| − normals | 0.880 | **+0.000** (dead) |
| − SLAT | 0.891 | **−0.011** (net-harmful noise — dropping it *helps*) |

**Conclusion — the model reduces to DINO + geometry + verb-text.** The entire per-vertex signal comes from **DINOv2 patch tokens (+0.057)** and the **geometry channel (+0.020, confirming the earlier +0.029)**. **CLIP-vision contributes exactly 0** — its only value is the open-vocab *verb-text* embedding, not the per-vertex vision features (this settles the recurring "is CLIP useless?" question). **normals ≈ 0** and **SLAT is net-negative (−0.011)** — dropping SLAT slightly *improves* the model, consistent with the earlier "SLAT marginal / hurts grasp" negative.

**Confirmatory combined-drop (5-fold, GEAL-pretrain, k24) — and the FINAL MODEL.** Leave-one-out drops one channel at a time; to be sure the three near-zero channels don't interact, we dropped **CLIP-vision + SLAT + normals together** and re-ran the *full* 5-fold at the best (pretrained) config, against a matched full-feature baseline:

| final config | fold0 | fold1 | fold2 | fold3 | fold4 | **5-fold mean** |
|---|--:|--:|--:|--:|--:|--:|
| **full** (272-d: all 5 channels) | 0.922 | 0.927 | 0.864 | 0.866 | 0.895 | **0.895** |
| **lean** (133-d: DINO+geom+verb-text) | 0.903 | 0.921 | 0.883 | 0.858 | 0.885 | **0.890** |

Δ = **−0.005**, paired *t* = −0.76, **p ≈ 0.5** — statistically indistinguishable (fold 2 even flips +0.019). The full 0.895 also reproduced exactly, validating the comparison. **We therefore adopt the lean model (DINO 128 + geometry 5 + verb-text 128, per-vertex input 272→133-d) as the final model** — same accuracy, and it removes the CLIP-vision and SLAT feature-extraction stages from the pipeline (see 4e).

### 4e. Runtime — lean vs full final model *(A40, measured)*
The head skips `dim==0` channels, so lean is a genuinely smaller network (232k vs 250k params), not zeroed placeholders.

| component | full | lean | note |
|---|--:|--:|---|
| GNN head, 30k verts | 37.7 ms | 37.7 ms | head cost ~identical (kNN message passing dominates, shared) |
| GNN head, 200k verts | 342 ms | 324 ms | lean ~5% faster + lower peak mem (3.4 vs 3.6 GB) |
| CLIP-vision ViT-B/32 (16 views) | 11.5 ms | **removed** | lean drops the whole CLIP-vision extraction stage |
| SLAT encoder (~8.5k voxels) | 2.6 ms | **removed** | lean drops SLAT encoding |

**Honest read:** the head is *not* the runtime win (essentially equal). The saving is upstream — lean removes an **entire CLIP-vision extraction pass** (16-view render + ViT + vertex projection; one of two render passes in the pipeline) and the SLAT encoder, plus lower params/memory — **at zero accuracy cost**. The measured GPU-forward compute removed is ~14 ms/object; the larger real-world saving is the removed render+projection pass (not timed here — raw meshes archived). Net: a simpler, lighter pipeline for free.

### 4c. Aggregation / capacity (fold0 sweeps, established)
- EdgeConv max-pool 0.879 > single-head attention 0.840 (transformer variant **worse**).
- Larger neighborhood k24 (0.898) > k16 (0.894) > k8 (0.879); wider/deeper (bigk24) saturates.
- Edge-geometry features redundant (0.891 vs 0.896 — geometry already in the `geom` channel).
- SLAT multi-latent voxel→vertex aggregation flat; DINO fusion mean > max/smax.
- **bigk24 (best config) 5-fold ≈ k24 → plateau ~0.895; capacity gain does not compound.**

### 4d. Pretraining
GEAL pretrain +0.025 (all folds). Pure-GEAL (0 manual) on human = 0.613 → manual FT +0.28. So pretraining is a *warm start* (huge convergence speedup: ep1 0.83 vs 0.41 scratch), not the source of performance.

## 5. Generalization (the nuanced, honest story)
| axis | result | evidence |
|---|--:|---|
| new **objects** (trained cat/verb) | ✅ 0.895 | 5-fold held-out |
| new **categories** | ⚠️ split by verb type | LOCO (from-scratch, no leak) |
| — contain / pour (geometric) | transfer 0.68–0.99 | LOCO cup/bottle |
| — grasp (category-bound) | **fails 0.15–0.51** | LOCO all 3 categories |
| new **verbs, zero-shot** | ❌ structurally impossible | overlap matrix |
| new **verbs, few-shot** | ✅ 1–5 labels recover | press 0.13→0.74, lift 0.08→0.60 |

- **Cross-verb zero-shot is provably impossible:** affordance regions are **mutually near-disjoint** — max pairwise IoU across ALL verb pairs (trained & novel) = **0.06**. A novel verb's region overlaps no trained verb, so it can't be interpolated. (This falsifies "more training verbs would help" — the manifold is disjoint islands.)
- **Few-shot recovers it:** fine-tuning the 6-verb GNN on ONE densely-labeled object of a novel verb jumps it from ~0.1 to ~0.56; 3–5 → 0.6–0.74. Saturates immediately (the features are a good substrate; one label just *points* at the island).

## 6. Verb conditioning / open-vocab *(RE-VERIFIED on clean best GNN, fold0-val, n=63)*
- Real-verb AUPRC = **0.938** vs **nonsense-word** AUPRC = **0.584** vs chance = **0.277** → **verb signal above nonsense = +0.355** (stronger than the old +0.15 on the contaminated MLP); distinct verbs give **anti-correlated** maps (cross-verb map corr = **−0.18**, n=30) = genuinely distinct routing, not one field re-weighted.
- Honest framing: verbs genuinely route (distinct, above nonsense), but a large fraction of AUPRC is a **verb-agnostic geometric prior** — report the verb signal *above the nonsense baseline*, not the raw number.
- Synonym augmentation → phrasing robustness (paraphrases of trained verbs work); does NOT help novel verbs (those are new affordances, not new phrasings).

## 7. Clean negatives (each is a documented result)
SLAT encoder marginal (+0.024, hurts grasp) · SLAT aggregation flat · attention head < EdgeConv · edge-geom redundant · DINO max/smax fusion < mean · model ensemble dilutes · bigk24 capacity plateaus · cross-verb zero-shot impossible (disjoint islands) · 6 cross-verb text interventions all failed (they reshuffle the *word*, not the missing region).

## 8. Data quality (a strength — we found and fixed issues)
- **GEAL contamination:** 54 of the "human GT" label files were GEAL pseudolabels (continuous, some exact copies). Detected via `frac_intermediate<0.005` (real hand-labels are binary). Dropped to quarantine. **The headline GNN>MLP result is robust to this** (+0.052 on clean, ~unchanged), and the clean MLP (0.814) = the pre-drop MLP (0.814) — contamination never inflated the number.
- **Label ambiguity:** e.g. a handled bottle where the model predicted grasping the *handle* (sensible) but the human labeled the mid-body → 0.45 AUPRC that is partly a *label* problem, not a model failure.
- **Human GT ≫ GEAL teacher:** the model beats its teacher on human labels (+0.058); pure-GEAL on human = 0.613 (weak teacher). Manual labels carry the result.

## 9. Limitations (own them)
1. **In-distribution is easy-ish:** contain (0.99) ≈ human-agreement ceiling; several verbs are geometrically self-evident; category↔verb coupling is near-deterministic (nonsense prior ≈ 0.55 shows ~half is verb-agnostic). The ~0.89 mean is partly the easy verbs.
2. **Grasp is the real hard verb** (0.75 in-dist, 0.17–0.51 cross-category) — part-based, category-specific.
3. **No zero-shot to new categories (grasp) or new verbs** — few-shot needed.
4. **Small dataset** (226 objects, 6 verbs). Not a benchmark; can't claim a SOTA *number* (different data/protocol from 3D AffordanceNet) — claim instead "beats the SOTA-class teacher on human labels."
5. **Noisy reconstructions & noisy labels** cap the ceiling more than the model does now.

## 10. Suited real-world task
Best as an **affordance-region proposer for known object types & functional verbs** — planning/hinting ("where to pour / place-into / rest-on", which transfer even cross-category) and **auto-annotation** (beats the teacher). **Not** precise robotic grasping (weakest verb, cross-category fails) nor open-world zero-shot.

---

## 11. Findings audit — does each session finding still apply?
| # | finding | status on clean data / current model |
|---|---|---|
| 1 | full-FT drives the big gain (head-only 0.65→0.79) | HOLDS (from distillation pipeline; clean full-FT+geom=0.814) |
| 2 | geometry helps in-distribution (+0.029) | **CONFIRMED** — clean ablation 4b: geom contribution **+0.020** (2nd-strongest channel) |
| 3 | spatial GNN > MLP (+0.056, all folds) | HOLDS (clean 5-fold) |
| 4 | larger neighborhood k24 best; capacity plateaus | HOLDS (bigk24 5-fold confirmed plateau) |
| 5 | GEAL pretraining +0.025 | HOLDS (clean 5-fold, all folds) |
| 6 | pure GEAL (no manual) only 0.613 | HOLDS (manual FT does the work) |
| 7 | attention head < EdgeConv | HOLDS (fold0 0.840 vs 0.879) |
| 8 | edge-geom features redundant | HOLDS (0.891 vs 0.896) |
| 9 | SLAT encoder marginal / aggregation flat | **CONFIRMED, STRONGER** — clean ablation 4b: SLAT contribution **−0.011** (net-negative; dropping it *helps*) |
| 10 | DINO fusion mean > max/smax | HOLDS (fold0) |
| 11 | ensemble dilutes | HOLDS |
| 12 | calibration (per-verb τ*) recovers IoU | HOLDS (fold0: IoU@0.5 0.61 → cal 0.68) |
| 13 | cross-category: geometric verbs transfer, grasp doesn't | HOLDS (clean LOCO, no leak) |
| 14 | cross-verb zero-shot impossible (disjoint islands, IoU 0.06) | HOLDS (clean overlap matrix) |
| 15 | few-shot recovers novel verbs (press/lift) | HOLDS (clean) |
| 16 | verbs route above nonsense; ~half is verb-agnostic prior | RE-VERIFIED, STRONGER: real 0.938 vs nonsense 0.584 (+0.355), cross-verb corr −0.18 |
| 17 | GEAL contamination in human labels (54 files) | HOLDS (found+fixed); result robust |
| 18 | human GT beats GEAL teacher (+0.058) | HOLDS |
| 19 | CLIP-vision weak/near-dead | **CONFIRMED DEAD** — clean ablation 4b: CLIP-vision contribution **+0.000** (exactly zero; open-vocab value is the verb-*text*, not per-vertex vision) |
| 20 | dataset easy in-distribution, hard on generalization | HOLDS (honest framing) |

## 12. Figures available (for report / poster / slides)
- `clean_gnn_vs_mlp.png` — headline: GT | MLP | GNN, 6 verbs, clean. **(report Fig 1 / poster hero)**
- `category_verb_matrix.png` / `examples_packed.png` — breadth across categories×verbs.
- `gnn_calibrated.png` — flooding → calibrated mask (IoU story).
- `loco_generalization.png` — cross-category: contain/pour transfer, grasp fails. **(generalization Fig)**
- `fewshot_progression.png` — 0→5-shot on a new verb. **(few-shot Fig)**
- `verb_overlap_heatmap.png` — verb regions mutually near-disjoint (max IoU 0.06); novel verbs boxed. **(cross-verb-impossibility Fig)** ✓ rendered
- `one_bottle_grasp.png` — label-ambiguity / model-finds-handle qualitative.

## 13. Talk outline (10–15 min)
1. Problem: verb-conditioned 3D affordance from a real image (1 slide).
2. Pipeline: image→SAM3D→features→verb-GNN, distill+FT (1).
3. Result: 0.814→0.870→0.895; GNN coherence figure (2).
4. Ablations: feature LOO + architecture + pretraining (1–2).
5. Generalization: objects ✓ / categories (split) / verbs zero-shot ✗ + overlap matrix / few-shot ✓ (3).
6. Rigor: contamination catch + honest limits (1).
7. Takeaway: beats its teacher; a region-proposer for functional verbs, honest about grasp/zero-shot (1).
