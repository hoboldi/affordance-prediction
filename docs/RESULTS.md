# Verb-conditioned affordance head — results summary (2026-06-24)

## ★ HEADLINE (2026-06-28): HUMAN LABELS BEAT THE GEAL TEACHER — cross-validated, significant
Measured vs **HUMAN GT** (`eval_human_gt.py`, trained-verb mean AUPRC). GEAL-agreement is misleading (teacher-mimicry), so all numbers are vs human. **Head-only fine-tuning** = freeze CLIP+DINO backbone, train verb_proj+out on human labels.

**5-fold cross-validation over all 225 labeled objects** (each object scored by the fold model that held it out; ~180 train/fold) — the trustworthy headline:

| verb | n | FT | GEAL | gap |
|---|--:|--:|--:|--:|
| contain | 157 | 0.748 | 0.488 | **+0.260** |
| sit | 29 | 0.817 | 0.534 | **+0.283** |
| pour | 63 | 0.762 | 0.737 | +0.025 |
| move | 29 | 0.705 | 0.651 | +0.055 |
| display | 37 | 0.543 | 0.669 | **−0.126** |
| grasp | 51 | 0.321 | 0.472 | **−0.151** |
| **trained-mean** | | **0.650** | **0.592** | **+0.058** |

**Bootstrap 95% CI on the gap = [+0.030, +0.085], P(FT>GEAL)=1.00.** The win is statistically solid.

### UPDATE (2026-07-03) — SLAT encoder, ensemble, full-FT (5-fold CV vs human; detail in docs/overnight_results.md)
Learned EdgeConv encoder over SAM3D's structured latent (`models/slat_encoder.py` + `mlp_slat.py`) replacing the flat 8-d SLAT channel.
- **Ensemble = confirmed free win:** avg(baseline + SLAT-encoder logits) = **0.724 vs 0.650 = +0.075, CI [+0.058, +0.091], P=1.0.** Lifts contain/display, recovers pour/move, pushes **grasp to 0.415 (+0.093)**. Render `outputs/renders/slat_gains.png`.
- SLAT-encoder head-only alone = +0.029 (marginal): strong contain/display, regresses pour/move.
- **★ full-FT (unfreeze trunk) = biggest signal:** 3-fold CV **0.823 (+0.17)**, holds across folds, **grasp 0.594** (the verb nothing else cracked). **CONFOUND unresolved:** the 0.650 baseline was *head-only*, so the gain may be full-fine-tuning itself, not the SLAT encoder — baseline-full-FT isolation check is GPU-gated/pending. Either way **full-FT ≫ head-only-FT** is likely a major finding.
- **full-FT is rigorously characterized (CPU stats, 07-03), confound aside:** bootstrap **95% CI [0.800, 0.843]** on the 0.823; gain over head-only **+0.160, CI [+0.134, +0.185], P=1.0**; **train→held-out gap only +0.043** (not memorizing — per-verb ≤0.06 except grasp +0.122); **no leakage** (0 exact-object, 0 same-source-version across all folds); per-fold means 0.834/0.843/0.790 (stable). So 0.823 is real, tight, and generalizes; the *only* open question is SLAT-encoder-vs-trunk-unfreezing.

- **Significance matters here:** a single 172/58 split gave a *misleading* +0.020 (CI [−0.041,+0.077], P=0.75 — not significant), because the small val had only n=7 sit/move. Scoring every object via 5-fold CV (sit/move on 29, grasp on 51) is what made the result trustworthy. **Always CV / bootstrap — single-split deltas at this scale are within noise.**
- **The win is concentrated:** FT dominates the high-volume, well-rendered verbs (**contain +0.26, sit +0.28**) but **LOSES grasp (−0.15) and display (−0.13)** to GEAL — exactly the coverage-limited verbs (handles/screens on under/occluded surfaces). GEAL's full-geometry PointNet++ has no visibility hole there.
- **Coverage is the diagnosed limiter for the loss verbs:** ~36% of vertices are invisible to the 25–60° upper-hemisphere camera policy; the model is near-blind there (AUPRC ~chance), and grasp's missing positives are 64% down-facing. Inpainting confirmed grasp is recoverable (+0.09) but head-only FT on guessed features can't net-convert it (frozen trunk). → real fix = **wider camera band + full retrain (Tier 2)**; interiors (contain/sit occlusions) → **geometry fallback (Tier 3)**.
- **Geometry trunk does NOT help under human-FT:** geom-trunk + human-FT = 0.446 (≪ plain). Helps geometric verbs, collapses semantic ones. Use geometry only as a targeted interior fallback, not a global trunk.
- **Caveats / open work:** all wins are vs our own GEAL teacher on our own human labels — **not yet benchmarked against recognized baselines** (required to claim field-level SOTA). Project goal is max accuracy (real-time infeasible — SAM3D dominates latency). Full detail in `docs/overnight_results.md`.

---

All numbers below are **per-verb AUPRC** from `eval_verb_conditioning.py` (~50 multi-verb objects, vs **GEAL** — pre-human-GT era), which is
**deterministic** and is the headline metric. Training-val *aggregate* AUPRC is misleading (high aggregate
≠ high per-verb) and is not used for conclusions. All models share: clean multi-view render + CLIP +
per-vertex DINOv2 + no contrastive loss + SLAT(8) + normals; trained on GEAL pseudolabels.

## Final recipe
- **Closed-vocab (best overall):** **cross-attention** verb head + learned verb embedding → **0.484**.
- **Open-vocab:** **concat** conditioning + CLIP-**text** verb embedding → **~0.42** (best seed 0.424).
  Handles arbitrary verb phrases; ~0.06 below the closed champion = the price of generality.

## Comparison
| model | vocab | conditioning | per-verb | corr | note |
|---|---|---|---|---|---|
| closed cross-attn | closed | cross_attn | **0.484** | −0.50 | best overall |
| base_l0 | closed | concat | 0.459 | −0.34 | |
| v15 | open | concat | **0.424** | — | best open-vocab (lucky seed) |
| ov_concat_nc | open | concat | 0.394 | +0.04 | reproduction (same config as v15) |
| ov_concat_bighead | open | concat + deep verb-proj + trunk 512/256/128 | 0.395 | +0.04 | "better head" — **no gain** |
| ov + xattn (shallow) | open | cross_attn | 0.378 | **+0.49** | cross-attn collapses verb distinctness |
| ov + xattn (bighead) | open | cross_attn big | ~0.20 | — | much worse |

## Key findings
1. **Stacked wins (baked into all rows):** drop contrastive (+9%), DINO dual-channel (+57% over CLIP-only),
   feature-render overhaul (+9%).
2. **cross-attn is a closed-vocab-only win.** On open-vocab (CLIP-text verbs, cos ~0.82) it cannot form
   separable verb queries → distinctness collapse (corr +0.49). Open-vocab must use **concat**.
3. **Head capacity is not a lever.** Deeper verb projection + wider trunk raised *aggregate* (0.461 vs 0.454)
   but left *per-verb* flat (0.395 ≈ 0.394). It only redistributed across verbs (better `move`, worse `pour`).
4. **Open-vocab ceiling ≈ 0.40 ± 0.02 (seed variance).** v15 (0.424) and two same-config reproductions
   (0.394, 0.395) differ only by seed — a reliable number needs multi-seed averaging.
5. **Eval is deterministic** (same checkpoint → identical per-verb); the earlier "noise" was two different
   checkpoints. So the closed cross-attn > base_l0 gap (0.484 vs 0.459, +0.025) is real.

## Open levers (deferred / untested)
- Finer DINO patches (32×32, `--dino_image_size 448`); more data (uncap categories); text-encoder LoRA;
  mesh-graph / neighborhood refinement.
- **Human-eval set** (`scripts/mesh_painter.py` → `human_gt_labels/`): every number above is vs the GEAL
  teacher; a small human-labelled set is the honest validation and the only way to push *past* GEAL.

## UPDATE 2026-06-26 — open-vocab NEW BEST: GNN + 3D-geometry + verb-in-backbone = 0.525 per-verb
Compound of: chunked/checkpointed EdgeConv GNN over a kNN mesh graph + 5-ch per-vertex geometry (height/concavity/curvature/normal-up/radial) + verb concatenated into the backbone (not just head).
| model | vocab | per-verb | corr | note |
|---|---|---|---|---|
| **GNN+geom+verb_in_backbone** | open | **0.525** | 0.090 | NEW BEST; contain 0.570, beats closed champ |
| closed cross-attn | closed | 0.484 | -0.50 | prior overall best |
| finer-DINO MLP | open | 0.455 | 0.002 | prior open-vocab best |
| MLP+geom | open | 0.419 | 0.619 | geometry helps contain but collapses verbs |
| GNN-alone | open | 0.384 | 0.940 | verb collapse (verb only at head) |
Key findings: (1) geometry (concavity) is the containment signal — contain 0.262->0.570 across the stack; (2) a relational backbone needs the verb injected into it (head-only -> collapse corr 0.94); (3) geometry must be verb-GATED (MLP+geom collapsed corr 0.62) — the GNN-with-verb-in-backbone does this, keeping geometry's gains AND verb distinctness.

## CORRECTION 2026-06-26: '0.525 NEW BEST (GNN+geom)' above is GEAL-AGREEMENT, not human-correctness.
Human-GT eval re-ranked (overall mean vs human): MLP(finer-DINO) 0.545 > GEAL teacher 0.533 > GNN+geom 0.437. The GNN+geom GEAL-win was teacher-mimicry (contain 0.570-vs-GEAL but 0.435-vs-human). REAL current best vs human = finer-DINO MLP (outputs/ov_concat_finepatch). See overnight_results.md human-GT leaderboard.

## Human-GT fine-tune (overnight 2026-06-27): head-only FT = NO GAIN; full FT blocked by shared-box contention.
Held-out val (vs human): base MLP 0.578 trained; head-only FT 0.579 (flat, overfits). Full FT pending a free GPU window. Conclusion: distillation/FT capped at the teacher on 120 in-distribution objects; bottleneck = label quantity -> active learning + held-out-object test.
