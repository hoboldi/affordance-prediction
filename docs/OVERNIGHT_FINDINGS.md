# Overnight autonomous run — findings log

_Started 2026-07-10 evening, completed ~00:20 2026-07-11. Agent worked through the promising-directions list._
_Constraints respected: nvidia-smi first, free GPU / CPU only, no OOM, no preemption, commits local-dev only, figures→final_figures. Box left clean (no running jobs)._

---
## ☀️ MORNING SUMMARY (read this first)

**6 experiments run, all complete. Two change the paper; two are clean negatives that validate current choices.**

1. **⭐ Two-class finding needs a correction (bootstrap CIs).** The **DINO/appearance axis is statistically
   robust** (Appearance drop-DINO +0.240 [0.182,0.296] vs Geometric +0.045 [0.033,0.058] — cleanly separated).
   But the **geometry axis is NOT** — it's driven by lift (n=12) and washes out item-weighted (Geo geom +0.041 ≈
   App +0.040). **Action: reframe §4.4 — lead on DINO-reliance; state geometry cautiously; drop "both axes split
   by class."** This is the most important overnight result.

2. **⭐ Open-vocab failure is CLIP-specific (new analysis nugget).** CLIP text places only 2/8 verb synonyms
   nearest their verb; **all-mpnet-base-v2 gets 8/8**. Swapping the verb encoder to mpnet is **free on exact
   verbs (0.872 ≈ 0.863)** and **rescues polysemy partially** (decant 0.10→0.51) — but does NOT fully solve
   open-vocab end-to-end (click/elevate still fail; −0.29 aggregate). **Action: add to limitations/analysis —
   the CLIP text-geometry bottleneck is real and partly fixable; full open-vocab = encoder+augmentation (future).**

3. **noboth ablation closed** — measured drop-both = **0.173** (was extrapolated ~0.18). Ablation table now fully
   measured. **Action: replace the extrapolated cell.**

4. **DINOv2-large: no gain** over base at matched 128-d (0.860 vs 0.872). **Validates the lean DINO-base choice.**

5. **Scratchpad persisted** to `experiments/cv_harness/` (143 files) + all overnight scripts in `experiments/`.

**Nothing committed** (per your local-dev rule — left staged for your review). New checkpoints:
`gnn_leanabl8_noboth_fold{3,4}`, `gnn_mpnet8_fold0`, `gnn_dinolg128_8_fold0`. New scripts under `experiments/`.

**Suggested next when you're back:** (a) apply the §4.4 bootstrap correction + the noboth cell to POSTER.md/paper;
(b) decide if the mpnet open-vocab nugget goes in the paper's analysis; (c) the poster wording fixes still pending.

---
## Detail log

## Plan
1. [done] Persist ephemeral scratchpad → `experiments/cv_harness/` (143 files)
2. [running] #1 Text-encoder geometry probe — does a better encoder place decant/click/elevate correctly?
3. [pending] #4 Bootstrap CIs on two-class ablation deltas
4. [running] Finish `noboth` (folds 3,4) → measured DINO+geom cell
5. [pending] #2 Cheap trained baseline (linear probe on DINO+geom)
6. [pending] #3 DINOv2-large on one fold (weak appearance verbs) — GPU-permitting

## Environment scouted
- GPU0 free (0%), GPU1 21GB/4% (other user). HF reachable. DINOv2-large cached (→ #3 needs no download).
- Cached text models: clip-vit-base-patch32, roberta-base. Will fetch sentence-transformers for the probe.

---

## Results

### 1. Scratchpad persisted ✅
143 files → `experiments/cv_harness/` (harness `train_gnn_cv.py`, CV splits `cv_fold{0-4}.json`, `manifest.cv.jsonl`, all eval + run scripts). NOTE: internal SCR paths in these still reference the `/tmp` scratchpad (which is currently alive); update to `experiments/` to run after `/tmp` is wiped.

### 2. #1 Text-encoder geometry probe ✅ — BIG RESULT
The open-vocab failure is **CLIP-specific, not fundamental.** Nearest-verb accuracy over 8 held-out synonyms:

| encoder | acc | hard-word rescues (decant/click/elevate) |
|---|--:|--:|
| CLIP-ViT-B/32 (current) | 2/8 | 0/3 |
| **all-mpnet-base-v2** | **8/8** | **3/3** |
| all-MiniLM-L6-v2 | 6/8 | 2/3 |
| roberta-base (raw) | 1/8 (degenerate) | 0/3 |

all-mpnet places every held-out synonym — including CLIP's 3 failures — nearest its correct verb.
→ **Follow-up launched: retrain the head with mpnet verb embeddings (fold0, from scratch, 30ep, GPU1),
then run the held-out synonym test.** If held-out synonyms now track exact-verb AUPRC (vs CLIP's −0.30),
open-vocab reopens as a positive result. Script: `experiments/encoder_probe.py`.

### 3. noboth (DINO+geom removed) — DONE ✅ measured drop-both cell
**5-fold held-out clean macro = 0.173** (was extrapolated ~0.18 — confirmed). Per-verb: contain 0.451
(weak structural prior from graph alone), all others 0.08–0.17 (≈chance). → paper ablation table now has a
measured floor: removing both per-vertex feature channels collapses the model to ~chance. Checkpoints
`gnn_leanabl8_noboth_fold{0-4}.pt`; eval `experiments/eval_noboth.py`.

### 4. mpnet-verb retrain (fold0) — DONE ✅ (nuanced, honest)
Swapped CLIP-text → all-mpnet-base-v2 as the verb encoder (from scratch, 30ep, mpnet 768-d).
- **Exact-verb macro 0.872** ≈ CLIP lean8 0.863 → **the encoder swap costs nothing** (even from scratch).
- **Held-out synonyms −0.29** (0.872→0.583): MIXED. Wins — **decant 0.10→0.513** (polysemy partially rescued),
  keep-inside 0.912, showcase 0.915, reposition 0.813, seize 0.650. Losses — click 0.333, elevate 0.094,
  rest-on 0.434 still fail (even SEEN raise/store regressed).
- **Conclusion:** the probe's nearest-verb 8/8 was necessary but NOT sufficient — nearest-among-8 ≠ landing
  in the trained region. A stronger encoder rescues some polysemous words at zero exact-verb cost, but
  end-to-end open-vocab remains only partly solved. Full fix likely = encoder + augmentation (future work).
- **Paper value:** analysis nugget for the limitations section — the CLIP text-geometry bottleneck is real
  and partly fixable, quantified. NOT a headline result. Open-vocab thread CLOSED (not over-chasing).
- Artifacts: `experiments/{encoder_probe,precompute_mpnet_verbs,synonym_test_mpnet_fold0}.py`,
  ckpt `gnn_mpnet8_fold0.pt`, emb `experiments/verb_emb_mpnet.pt`.

### 5. #4 Bootstrap CIs on two-class deltas — DONE ✅ IMPORTANT (tempers the finding)
| class | drop-DINO Δ [95% CI] | drop-geom Δ [95% CI] |
|---|---|---|
| Geometric (n=264) | +0.045 [0.033, 0.058] | +0.041 [0.030, 0.052] |
| Appearance (n=90) | **+0.240 [0.182, 0.296]** | +0.040 [0.015, 0.065] |

- **DINO appearance-specificity: ROBUST** — Appearance +0.24 vs Geometric +0.045, CIs cleanly separated (~5×).
- **Geometry geometric-specificity: NOT supported** — Geo +0.041 ≈ App +0.040, CIs overlap. The claim was
  driven by lift (n=12, −0.31); item-weighted it washes out against contain (n=142, −0.02).
- **PAPER ACTION:** lead the two-class finding on the DINO/appearance axis (statistically robust); state the
  geometry axis cautiously (lift-driven, not class-separating). DROP "both feature axes split by class."
- Script `experiments/bootstrap_ci_finding.py`. (Item-level bootstrap; object-level would be even more conservative.)

### 6. #3 DINOv2-large (weak appearance verbs) — first attempt snagged, relaunched
- Snag: `vertex_dino_large.pt` already existed at **256-d** (prior run); `--skip_existing` left 229 objects at
  256-d, train failed on dim mismatch (model wants 128). NOTE FOR HANDOVER: a 256-d DINO-large feature set
  already exists on disk (could train a from-scratch dino_dim=256 variant if desired).
- Fix: re-precomputed to fresh `vertex_dino_lg128.pt` @128-d (238 objects), warm-started fold0 train.
- **Result: NO improvement.** dinolg128 fold0 tm **0.860** vs lean8 (DINO-base) fold0 **0.872** (−0.012).
  Appearance verbs did not lift (grasp 0.759, display 0.889, press 0.809). At matched 128-d, DINOv2-large
  gives no gain over base — the random projection to 128 likely bottlenecks large's extra capacity.
- **Conclusion:** validates the lean DINO-base-128 choice; do NOT upgrade the backbone. (A 256-d from-scratch
  variant could preserve more of large — deferred; 256-d `vertex_dino_large.pt` already on disk.)
  Ckpt `gnn_dinolg128_8_fold0.pt`.

---
## DAY 2 — paper-hardening (projection, fidelity, headline-protection)

### 7. Projection: random vs PCA — DONE ✅ critique defused
Random-128 (current) fold0 **0.872** vs PCA-128 fold0 **0.865** → **comparable** (random marginally ahead).
The "unusual random projection" choice is validated. Paper line: *"random projection; PCA performs comparably
(0.865 vs 0.872), we use random for simplicity."* DROP the Johnson–Lindenstrauss editorializing.
Scripts: `experiments/{fit_apply_pca.py}`, ckpt `gnn_dinopca128_8_fold0.pt`.

### 8. Reconstruction fidelity vs CO3D ground truth — DONE ✅
GT clouds already on disk (`cmr2/source/{cat}/{seq}/pointcloud.ply`). Normalize+ICP(24 restarts)+Chamfer/F.
**n=86 of 238 CV objects have GT.** Mean **Chamfer 0.247 ± 0.112**, **F@0.05 0.397 ± 0.184**; category-dependent
(laptop/keyboard F≈0.5 best, handbag F0.26 worst). Caveats: subset with GT; unsegmented reference → **upper
bound on error**; normalized (relative) units; labels annotated ON reconstructions. Script `experiments/fidelity_chamfer.py`.

### 9. Fidelity × performance correlation — DONE ✅ (scenario #1, weak form)
Per-object F-score vs ReVerb AUPRC (n=70). **ALL: ρ=0.30 (p=0.01)**; GEOMETRIC ρ=0.26 (n.s.); APPEARANCE ρ=0.19 (n.s.).
→ **Weak but real positive; fidelity explains <10% of variance; classes do NOT separate.** Claim: *affordance
largely robust to geometric infidelity — mesh must be plausible, not accurate* (justifies 2D→3D scaffold).
NOT scenario #2 (no class separation), NOT #3 (too weak). Caveats: noisy proxy attenuates → lower bound;
labels-on-reconstruction → robustness is vs same-mesh labels, not reality (reprojection = future work).
Script `experiments/correlate_fidelity_affordance.py`.

### 10. P1 headline-protection checks — DONE ✅ FINDING SURVIVES
- **Region-size confound (real but survived):** appearance regions smaller (median posfrac 0.15 vs 0.35).
  DINO-drop class coef 0.195 → **0.166 with size control, bootstrap CI [0.107, 0.226] excludes 0**;
  size-matched subset (n=159) Geometric **0.065** vs Appearance **0.187**. → split is NOT a size artifact.
- **Leave-one-verb-out:** App−Geo DINO gap **0.170–0.223** dropping any verb → not carried by one verb.
- **0.52 decomposition (check 3):** pretrained-only beats GEAL on **6/8** verbs (grasp −0.005, lift −0.005 excepted);
  broad, not 1–2-verb-driven → claim it with the grasp/lift caveat.
  Scripts `experiments/{checks_region_lovo.py, check_pretrained_perverb.py}`.
- **PAPER ADD:** §4.4 gains a region-size defense paragraph (the highest-value add — pre-empts the fatal confound).

### 11. GEAL inference inputs — CLARIFIED
GEAL at inference uses **point cloud + text affordance question ONLY**; its 2D Gaussian-splatting branch is
**training-only** (`src/labeling/geal_infer.py`). We feed it its native inputs (point cloud from our recon + verb
question) → the zero-shot comparison is **fair on inputs** (no withheld interaction image). Only asymmetry =
domain shift (trained on clean clouds, run on reconstructions). Strengthens the teacher-comparison framing.

### 12. Check A + Check B — DONE ✅✅ HEADLINE IS TASK-INTRINSIC (strongest result)
**Check A (inherited vs learned):** GEAL-distilled head (no human FT) DINO-drop: Geometric −0.00, Appearance
**+0.064**. Finetuned: Geometric +0.06, Appearance **+0.245**. → the appearance-DINO reliance is **LEARNED in
human finetuning (~4×), not inherited from GEAL**.

**Check B (from scratch, NO teacher, 5-fold):** DINO-drop Geometric **+0.038**, Appearance **+0.220**, gap
**+0.181** — nearly identical to finetuned gap +0.185. LOVO gap 0.161–0.207 (robust). From-scratch absolute
0.928 geo / 0.840 app. → the two-class split is **TASK-INTRINSIC — reproduces with no teacher at full magnitude.**

**COMBINED VERDICT:** the appearance→DINO structure is learned from human affordance labels, independent of GEAL,
and intrinsic to the task (survives: region-size control, LOVO, teacher removal). Stronger than a held-out-verb
prediction — it's a causal/intervention result.

**PAPER ACTIONS:**
- §4.4 add: "split reproduces from-scratch with no teacher (gap +0.18 vs +0.19) → property of the task, not the
  distillation." Optional small table: DINO-gap across {from-scratch, distilled, finetuned}.
- **DELETE the "can't train without a teacher" limitation** — refuted (from-scratch 0.93 geo / 0.84 app). Reframe
  teacher as a label-efficiency bootstrap (not required at full data).
Ckpts `gnn_scratch8_fold{0-4}.pt`, `gnn_scratchnodino8_fold{0-4}.pt`. Scripts
`experiments/{check_pretrained_class_dino.py, eval_scratch_dino.py}`.

### 13. Label-efficiency WITH vs WITHOUT teacher — DONE ✅ Stage 1 is a finding
8-verb, fold0 val=48, best macro AUPRC:
| labels | scratch | pretrained | gap |
|--:|--:|--:|--:|
| 10 | 0.388 | 0.593 | **+0.205** |
| 25 | 0.683 | 0.776 | +0.093 |
| 50 | 0.832 | 0.843 | +0.011 |
| 100 | 0.861 | 0.873 | +0.012 |
| full(~190) | 0.878 | 0.876 | −0.002 |

**Distillation is a warm start: +0.21 at 10 labels, +0.09 at 25, ~0 by 50, zero at full.** The teacher buys
**label efficiency, not final accuracy.** Justifies Stage 1 honestly AND confirms (with Check B) that the
two-class finding is not a teacher artifact (teacher inert at full data). Clean 2-line figure available.
Caveat: single fold (trend robust; 2-3 more folds → error bars if wanted). Scripts
`experiments/parse_labeleff.py`, ckpts `gnn_le8_{pre,scr}_{10,25,50,100,0}_fold0.pt`.

_(more results appended below as they complete)_
