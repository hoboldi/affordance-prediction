# Overnight results — started Tue Jun 23 10:00:41 PM UTC 2026
### STAGE0 v13_dinolarge  22:33
  contain     AUPRC=0.177 (n= 39)  pred_std=0.0802
  grasp       AUPRC=0.221 (n= 29)  pred_std=0.1896
  move        AUPRC=0.580 (n= 10)  pred_std=0.1666
  pour        AUPRC=0.562 (n= 29)  pred_std=0.1803
  sit         AUPRC=0.509 (n= 10)  pred_std=0.2037
VERDICT corr_learn=-0.419 iou_learn=0.000 std=0.1501

### STAGE0 v15_ovnc  22:35
  contain     AUPRC=0.322 (n= 39)  pred_std=0.1615
  grasp       AUPRC=0.339 (n= 29)  pred_std=0.1570
  move        AUPRC=0.521 (n= 10)  pred_std=0.1663
  pour        AUPRC=0.522 (n= 29)  pred_std=0.1800
  sit         AUPRC=0.414 (n= 10)  pred_std=0.1931
VERDICT corr_learn=0.006 iou_learn=0.000 std=0.1680

### base_l0  (--contrastive_weight 0)  02:53
  contain     AUPRC=0.297 (n= 39)  pred_std=0.1386
  grasp       AUPRC=0.360 (n= 29)  pred_std=0.1445
  move        AUPRC=0.551 (n= 10)  pred_std=0.1793
  pour        AUPRC=0.609 (n= 29)  pred_std=0.1832
  sit         AUPRC=0.477 (n= 10)  pred_std=0.2009
VERDICT corr_learn=-0.338 iou_learn=0.000 std=0.1598

### base_l03  (--contrastive_weight 0.3)  04:14
  contain     AUPRC=0.228 (n= 39)  pred_std=0.1074
  grasp       AUPRC=0.278 (n= 29)  pred_std=0.1266
  move        AUPRC=0.575 (n= 10)  pred_std=0.1712
  pour        AUPRC=0.610 (n= 29)  pred_std=0.1864
  sit         AUPRC=0.503 (n= 10)  pred_std=0.1894
VERDICT corr_learn=-0.428 iou_learn=0.000 std=0.1441

### base_crossattn_nocontrast (cross_attn, no-contrastive) 06:36
### cross-attn (stopped after ep20) — best.pt PER-VERB  08:39
  contain     AUPRC=0.302 (n= 39)  pred_std=0.1557
  grasp       AUPRC=0.382 (n= 29)  pred_std=0.1499
  move        AUPRC=0.602 (n= 10)  pred_std=0.1639
  pour        AUPRC=0.646 (n= 29)  pred_std=0.1846
  sit         AUPRC=0.490 (n= 10)  pred_std=0.1963
VERDICT corr_learn=-0.500 iou_learn=0.000 std=0.1655

  contain     AUPRC=0.301 (n= 39)  pred_std=0.1365
  grasp       AUPRC=0.370 (n= 29)  pred_std=0.1098
  move        AUPRC=0.590 (n= 10)  pred_std=0.1405
  pour        AUPRC=0.630 (n= 29)  pred_std=0.1805
  sit         AUPRC=0.475 (n= 10)  pred_std=0.2030
VERDICT corr_learn=-0.463 iou_learn=0.000 std=0.1467
### cross-attn (stopped after ep20) — last.pt PER-VERB
  contain     AUPRC=0.302 (n= 39)  pred_std=0.1557
  grasp       AUPRC=0.382 (n= 29)  pred_std=0.1499
  move        AUPRC=0.602 (n= 10)  pred_std=0.1639
  pour        AUPRC=0.646 (n= 29)  pred_std=0.1846
  sit         AUPRC=0.490 (n= 10)  pred_std=0.1963
VERDICT corr_learn=-0.500 iou_learn=0.000 std=0.1655

### ov_concat_nc (open-vocab concat, flagship) 16:41
  contain     AUPRC=0.308 (n= 39)  pred_std=0.1508
  grasp       AUPRC=0.305 (n= 29)  pred_std=0.1806
  move        AUPRC=0.397 (n= 10)  pred_std=0.1887
  pour        AUPRC=0.538 (n= 29)  pred_std=0.1851
  sit         AUPRC=0.420 (n= 10)  pred_std=0.1879
VERDICT corr_learn=0.036 iou_learn=0.000 std=0.1731

### ov_concat_bighead (open-vocab concat + deep_verb_proj + 512/256/128) 14:37
  contain     AUPRC=0.315 (n= 39)  pred_std=0.1223
  grasp       AUPRC=0.291 (n= 29)  pred_std=0.1607
  move        AUPRC=0.490 (n= 10)  pred_std=0.1530
  pour        AUPRC=0.462 (n= 29)  pred_std=0.1995
  sit         AUPRC=0.418 (n= 10)  pred_std=0.1443
VERDICT corr_learn=0.039 iou_learn=0.001 std=0.1556

### scale_25 (open-vocab concat, 25% = 240 train) 20:53
  contain     AUPRC=0.248 (n= 39)  pred_std=0.0994
  grasp       AUPRC=0.182 (n= 29)  pred_std=0.1158
  move        AUPRC=0.255 (n= 10)  pred_std=0.1019
  pour        AUPRC=0.317 (n= 29)  pred_std=0.1232
  sit         AUPRC=0.264 (n= 10)  pred_std=0.1009
VERDICT corr_learn=0.991 iou_learn=0.905 std=0.1098

### ov_concat_big (open-vocab concat, ~1436 data, early-stopped @ best ep13) 11:00
  contain     AUPRC=0.291 (n= 44)  pred_std=0.0831
  grasp       AUPRC=0.301 (n= 31)  pred_std=0.0932
  move        AUPRC=0.241 (n=  6)  pred_std=0.1107
  pour        AUPRC=0.561 (n= 37)  pred_std=0.1345
  sit         AUPRC=0.473 (n=  5)  pred_std=0.1206
VERDICT corr_learn=0.242 iou_learn=0.062 std=0.1036

### ov_concat_big — eval on 1180 eval set (fair comparison) 11:02
  contain     AUPRC=0.300 (n= 39)  pred_std=0.0951
  grasp       AUPRC=0.289 (n= 29)  pred_std=0.1263
  move        AUPRC=0.462 (n= 10)  pred_std=0.1408
  pour        AUPRC=0.589 (n= 29)  pred_std=0.1529
  sit         AUPRC=0.464 (n= 10)  pred_std=0.1443
VERDICT corr_learn=0.012 iou_learn=0.014 std=0.1252

### baseline ov_concat_big on BALANCED eval (n~80-120/verb) — reference for finer-patch comparison 13:22
  contain 0.302 (n=117) | grasp 0.285 (n=79) | move 0.472 (n=92) | pour 0.554 (n=87) | sit 0.522 (n=92) | mean 0.427 | corr 0.002

### FINER DINO 32x32 (ov_concat_finepatch) vs baseline — BALANCED eval 23:22 — VERDICT: WIN +6.6%
  verb     | finer32 | base   | delta
  contain  | 0.262   | 0.302  | -0.040  (regressed — interior/volume)
  grasp    | 0.330   | 0.285  | +0.045  (WEAK-VERB WIN)
  move     | 0.536   | 0.472  | +0.064
  pour     | 0.626   | 0.554  | +0.072
  sit      | 0.521   | 0.522  | -0.001
  MEAN     | 0.455   | 0.427  | +0.028  (+6.6%); corr -0.052 vs +0.002 (better distinctness)
  => new best OPEN-VOCAB (0.455); gap to closed champion 0.484 shrinks 0.057 -> 0.029. Finer patches sharpen surface/handle (grasp/pour/move up) but hurt interior (contain down).

### GNN ep12 (mid-training, stopped) on BALANCED eval 07:04 — VERDICT: verb-conditioning COLLAPSE
  verb     | GNN    | MLP    | delta
  contain  | 0.345  | 0.262  | +0.083  (generic saliency overlaps interior)
  grasp    | 0.320  | 0.330  | -0.010
  move     | 0.296  | 0.536  | -0.240  (needs distinct region -> craters)
  pour     | 0.486  | 0.626  | -0.140
  sit      | 0.471  | 0.521  | -0.050
  MEAN     | 0.384  | 0.455  | -0.071
  corr_learn 0.940 vs MLP 0.002  <-- predicts ~SAME map for every verb (collapse)
  ROOT CAUSE: verb concatenated only at shallow HEAD, after verb-agnostic EdgeConv backbone -> backbone saliency dominates, head can't re-localize. FIX: inject verb into the GNN backbone (per-vertex, before EdgeConv), like the MLP does at input.
  agg trajectory ep1-12: .266 .328 .357 .377 .372 .393 .423 .452 .457 .465 .469 .471 (~28min/ep, checkpointed). Stopped at ep12 (plateau).

### GEOMETRY-ONLY (MLP+geom ep4 best) on BALANCED 09:26 — contain WIN, verb partial-collapse
  contain 0.419 (MLP .262, +0.157 BIG) | grasp 0.333 (+.003) | move 0.473 (-.063) | pour 0.563 (-.063) | sit 0.306 (MLP .521, -0.215) | MEAN 0.419 vs MLP 0.455 (-0.036)
  corr_learn 0.619 vs MLP 0.002 -> geometry is verb-AGNOSTIC+dominant -> partial collapse. Helps geometry-aligned (contain) but hurts verb-specific (sit/move/pour). FIX=verb must GATE geometry (GNN verb_in_backbone+geom, next). Caveat: best.pt=ep4 early agg-peak; agg declined after (ep8 .473).

### *** GNN+geom+verb_in_backbone (best.pt ep12) on BALANCED 23:12 — NEW BEST, compound WIN ***
  verb     | GNN+geom | MLP    | MLP+geom | GNN-alone
  contain  | 0.570    | 0.262  | 0.419    | 0.345   (+0.308 vs MLP!! geometry+relational aggregation)
  grasp    | 0.373    | 0.330  | 0.333    | 0.320   (best-ever, +0.043)
  move     | 0.546    | 0.536  | 0.473    | 0.296   (recovered)
  pour     | 0.613    | 0.626  | 0.563    | 0.486   (held)
  sit      | 0.523    | 0.521  | 0.306    | 0.471   (recovered from MLP+geom collapse)
  MEAN     | 0.525    | 0.455  | 0.419    | 0.384   (+0.070 / +15% over MLP; BEATS closed-vocab champ 0.484)
  corr_learn 0.090 (MLP .002, MLP+geom .619 collapse, GNN-alone .940) -> verb_in_backbone FIXED the collapse.
  CONCLUSION: compound effect confirmed. geometry supplies signal + GNN aggregates relationally + verb_in_backbone keeps verbs distinct. Each piece necessary. agg trajectory ep1-18 peaked ep12 0.499 (best.pt=ep12 mature).


## *** HONEST LEADERBOARD vs HUMAN GT 2026-06-26 23:28 — GEAL ranking OVERTURNED ***
Per-verb AUPRC vs human_gt_labels (254/265 labels; 11 keyboards missing from manifest). In-distribution objects; press/lift/open are UNSEEN verbs.
| verb | GNN+geom | MLP(finer) | GEAL teacher |
|---|---|---|---|
| contain | 0.435 | 0.552 | 0.498 |
| display | 0.294 | 0.579 | 0.726 |
| grasp   | 0.299 | 0.293 | 0.424 |
| move    | 0.687 | 0.671 | 0.718 |
| pour    | 0.609 | 0.697 | 0.646 |
| sit     | 0.596 | 0.794 | 0.526 |
| TRAINED mean | 0.487 | **0.598** | 0.590 |
| unseen lift/open/press mean | 0.095 | 0.122 | **0.243** |
| OVERALL mean | 0.437 | **0.545** | 0.533 |

FINDINGS (the reason human GT was essential):
1. RANKING REVERSED. GEAL-agreement said GNN+geom 0.525 > closed 0.484 > MLP 0.455. HUMAN says MLP 0.545 > GEAL 0.533 > GNN+geom 0.437. The GNN+geom's GEAL win was TEACHER-MIMICRY: its contain 0.570-vs-GEAL is 0.435-vs-human (< MLP's 0.552). Optimizing GEAL-agreement steered us wrong.
2. TEACHER IS THE CEILING (empirical). Best model (MLP) trained-mean 0.598 ~= GEAL 0.590. Distillation matches, does not exceed.
3. OPEN-VOCAB to NOVEL verbs is weak. Models fail on open/press (CLIP-text extrapolation misses); GEAL's direct labels do better (unseen 0.243 vs model ~0.10).
IMPLICATIONS: use human GT as the selection+eval metric (not GEAL-agg); MLP(finer-DINO) is the real current best; to EXCEED the teacher, fine-tune on human GT (only way past the ceiling, now confirmed); re-test whether geometry helps when trained/selected vs human (GEAL-optimization may have hidden its real value).

### HUMAN VAL baselines (42-obj held-out, vs human) — FT reference 23:45
  MLP(ov_concat_finepatch): trained 0.578 | unseen 0.130 | overall 0.536
  GNN+geom(gnn_geom):       trained 0.521 | unseen 0.090 | overall 0.455
  GEAL teacher:             trained 0.551 | unseen 0.331 | overall 0.509
  => MLP best on val; GEAL strong on unseen (direct labels). FT must beat MLP 0.578 trained / 0.536 overall.

### HUMAN-FT head-only MLP seed0 — NO GAIN (head-only too limited) 00:09
  early-stop ep9; val_mean peaked ep1 0.427 then fell monotonically (overfit: train_bce 3.77->1.36, val down).
  best.pt(ep1) val: trained 0.578 overall 0.540 | last.pt(ep9): trained 0.579 unseen 0.094(base .130) overall 0.552.
  trained-mean FLAT vs base 0.578 (contain 0.582->0.623 but offset by other verbs; unseen DOWN). Frozen backbone can't adapt to human truth.
  ACTION: pivot to FULL fine-tune (--no_freeze_backbone, lr 3e-5). Skipped head-only seeds1-2/GNN as clearly unproductive.

## HUMAN-GT FINE-TUNE — interim verdict 2026-06-27 01:05 (partial: shared-box contention)
- HEAD-ONLY FT (MLP, frozen backbone): NO GAIN. val trained-mean flat 0.578->0.579, unseen DOWN .130->.094, overfits (train_bce falls, val_mean falls). Frozen features can't adapt to human truth.
- FULL FT (MLP, --no_freeze_backbone, subsampled 60k verts): could NOT complete tonight. Subsampling made it ~45s/ep when box idle, but the shared box got saturated by another user (GPU1 92% util / 18GB, load avg ~38) -> my run crawled >22min/ep -> killed to yield (shared-box policy). Retry queued for a free GPU window.
- STANDING CONCLUSION: on 120 in-distribution human-labeled objects, fine-tuning has not exceeded the GEAL teacher (head-only flat; full pending a free window). With the earlier leaderboard reversal (MLP real-best ~0.545 vs human; GEAL-best GNN+geom is human-worst), the bottleneck is LABEL QUANTITY/DIVERSITY + in-distribution objects, not the FT recipe.
- NEXT: active-learning to label high-value objects; a held-out-OBJECT human test; re-attempt full FT (+seeds) in a free GPU window; complete geometry-vs-human leaderboard.


## PHASE D SYNTHESIS — Human-GT fine-tune investigation 2026-06-27 02:40
GOAL: can fine-tuning on human GT exceed the GEAL teacher? Metric: held-out 42-obj val, per-verb AUPRC vs human.
VAL baselines: MLP 0.578 trained / 0.536 overall | GEAL 0.551 trained | GNN+geom 0.521.
RESULTS:
- HEAD-ONLY FT (frozen backbone): NO GAIN. trained-mean flat 0.578->0.579, unseen DOWN .130->.094, overfits (val_mean fell from ep1). Frozen features can't adapt to human truth.
- FULL FT (--no_freeze_backbone, subsampled 60k verts): could NOT complete. Repeatedly CPU-bound-stalled (proc 17 cores / GPU0 0% util / 0 epochs) under the loaded shared box (other user's sustained GPU1 job since ~00:30). 3 launches killed to yield. NOTE: two SEPARATE A40s (not MIG); the stall is the CPU/IO data path under box load, not GPU0 compute. Pending a free window.
VERDICT (tonight): on 120 in-distribution human-labeled objects, fine-tuning has NOT exceeded the teacher (head-only flat; full pending free box).
BIGGER PICTURE: GEAL-agreement is misleading (GNN+geom 0.525-vs-GEAL is human-WORST 0.437; MLP is real-best 0.545). Teacher is the empirical ceiling (best model ~= GEAL trained-mean).
BOTTLENECK: label quantity/diversity (120 in-distribution objs), NOT the FT recipe.
RECOMMENDATIONS: (1) ACTIVE-LEARNING labeling (high-uncertainty / GEAL-vs-model-disagreement / weak verbs) to maximize value/label. (2) HELD-OUT-OBJECT human test (objects never in GEAL training) for true generalization. (3) Re-run full-FT (+seeds) in a free GPU window to settle if unfreezing helps. (4) Novel-verb open-vocab weak (open/press ~0.1 vs GEAL 0.33) -> compositional/LLM-grounded verbs.

## GEOMETRY-vs-HUMAN leaderboard 06:42 — geometry HURTS vs human (full set, trained-mean)
  MLP base 0.598 (real best) > GEAL 0.590 > GNN+geom 0.487 > MLP+geom 0.417   [overall: MLP .545 > GEAL .533 > GNN+geom .437 > MLP+geom .430]
  => every geometry/relational addition that WON on GEAL-agreement DEGRADED human-correctness (overfit teacher quirks). Plain finer-DINO MLP is unambiguously best vs human.
  (gnn_finepatch eval skipped: old ckpt predates verb_in_backbone -> state_dict mismatch; it was the verb-collapsed GNN.)
FULL-FT: abandoned tonight after 4 attempts — subsampled full-backward still >13min/epoch even on a FREE box (perf issue on large meshes, needs mesh-decimation/batched-vertex engineering). Head-only negative + this geometry-hurts result make the verdict robust.
FINAL OVERNIGHT VERDICT: the simplest model (finer-DINO MLP) is the real best vs human; GEAL-agreement actively misled architecture choices; fine-tuning on 120 in-distribution objects does not exceed the teacher. NEXT (highest leverage): active-learning human labeling + held-out-object human test + (engineering) a fast full-FT path; treat human GT as the metric of record.

## CPU-ONLY TESTS (no model/GPU; run under box contention) 2026-06-27 20:34
TEST 1 — TEACHER ERROR MAP (GEAL pseudolabel vs HUMAN AUPRC per verb; low = GEAL wrong = headroom):
  lift 0.085 | open 0.282 | press 0.323 | grasp 0.472 | contain 0.488 | sit 0.534 | move 0.651 | display 0.669 | pour 0.737 | OVERALL 0.530 (410 pairs). GEAL worst on novel verbs + grasp/contain; best on pour.
TEST 2 — HUMAN cross-verb DISTINCTNESS (the ideal target): median corr -0.148, top10%-IoU 0.028 (232 pairs) — humans paint near-DISJOINT regions per verb. vs models: MLP 0.002 (closest), GNN+geom 0.09, MLP+geom 0.62, GNN-alone 0.94. -> MLP structurally right; geometry/GNN collapse genuinely off-target.
TEST 3 — CONTAIN UNDER-SEGMENTATION: GEAL marks containment at ~6.5-8.4x SMALLER region than humans, UNIFORMLY across all container categories (human posrate ~0.39-0.50 = full interior; GEAL ~0.06-0.075 = tiny patch). Massive systematic teacher bias.
IMPLICATION: clearest place to BEAT THE TEACHER = CONTAIN (biggest systematic GEAL error + most labels, 157) -> human-FT should win big on contain IF FT mechanism works. Then grasp + novel verbs (lift/press/open: huge GEAL error, few labels). Queued learning curve will quantify.

## GEOMETRY directly predicts HUMAN affordances (CPU test) 21:45 — REFRAMES 'geometry hurts'
Single raw geom-channel AUPRC vs human (best sign), per verb:
  contain normal_up 0.727 (concavity 0.606) | sit normal_up 0.694 | pour height_up 0.847 | move height_up 0.806 | lift height_up 0.578 | press normal_up 0.544 | grasp radial 0.434 (weakest) | display height_up 0.307
KEY: gravity-aligned geometry (height_up, normal_up) is HIGHLY predictive of HUMAN affordances; a SINGLE geom channel rivals/BEATS the trained models vs human on pour (0.85 vs ~0.70), contain (0.73 vs 0.55), move (0.81 vs ~0.67).
IMPLICATION: geometry is NOT useless. Earlier 'geometry hurts vs human' was GEAL-MEDIATED (model distilled GEAL's biased labels, entangling geometry). Geometry vs HUMAN is excellent => geometry + HUMAN supervision is PROMISING; RE-TEST geometry+GNN with human-FT (do not abandon). Caveat: single-channel best-sign (mild optimism), base rates vary per verb.

## NEW-SCALE val baselines (58-obj val, vs human) 06-28 04:29
  MLP base: trained 0.580 / unseen 0.077 / overall 0.532 | GEAL: trained 0.593 / unseen 0.243 / overall 0.534
  Crossover target for FT = beat GEAL 0.593 (and MLP 0.580) trained-mean.

## ★ LEARNING CURVE — HUMAN LABELS BEAT THE GEAL TEACHER 06-28 07:19
Head-only FT (freeze CLIP+DINO backbone, train verb_proj+out only) on human GT, trained-verb mean AUPRC vs HUMAN (58-obj val):
  #train-objects :   0(base)   50      100        | teacher
  trained-mean   :   0.580   0.593   **0.613**    | GEAL 0.593
  -> N=50 REACHES teacher parity (0.593); N=100 EXCEEDS it (+0.020). Monotonic rise. (n172 abandoned: box read 172-obj fine-DINO too slowly to finish epoch-1 within the 50min timeout — contention artifact, not a result.)
SLOPE ~ +0.013/50obj then +0.020/50obj (≈ +0.0003–0.0004 per labeled object) over 0→100; accelerating, not yet saturating at 100.
PROJECTION (linear, optimistic — needs n≥150 to confirm): ~50 labels = GEAL parity, ~100 = clearly beat teacher, ~200 ≈ 0.65, ~300 ≈ 0.69. Saturation point unknown (box-blocked above 100).
PER-VERB gain (base → head-FT@100, vs human): contain 0.552→0.619 (+0.067) | sit 0.717→0.779 (+0.062) | move 0.603→0.645 (+0.042) | pour 0.644→0.685 (+0.041) | display 0.556→0.550 (flat) | grasp 0.410→0.401 (flat). FT lifts the densely-labeled geometric/containment verbs; grasp/display (sparse: n=12/8) stay flat = next labeling priority. Open-vocab unseen verbs unaffected (lift 0.112/open 0.004/press 0.285, mean 0.134).

## GEOMETRY+HUMAN-FT verdict — geometry HURTS human-correctness (confirms GEAL-overfit illusion) 06-28 07:19
geom+human-FT@N=100 (head-only FT from outputs/mlp_geom GEAL-trained geom-trunk): trained-mean **0.446** — far BELOW plain head-FT@100 (0.613) and even below base (0.580).
  Per-verb: contain 0.662 (HIGHER than plain 0.619 — geometry DOES help the geometric affordance) but display 0.212 (COLLAPSED vs 0.550) — geometry tanks semantic verbs. Net strongly negative.
VERDICT: the GEAL-trained geometry trunk is a WORSE base for human-FT than plain finer-DINO. Geometry helps geometric verbs (contain/sit) but is useless/harmful for semantic ones (display/grasp); on balance plain CLIP+DINO wins. Confirms earlier "geometry/GNN gains were GEAL-overfit illusions." CAVEAT/confound: the geom trunk also uses COARSER (16×16) DINO, so part of the 0.613→0.446 gap is the DINO downgrade, not geometry alone — a clean test would add geom to the 32×32-DINO model. But the practical conclusion holds: do NOT route through the geom trunk for human-correctness.

## RECOMMENDATION (settled, 06-28)
1. **Drop GEAL as the target.** It is now the FLOOR, not the ceiling — head-only FT on ~100 human-labeled objects beats it (0.613 vs 0.593).
2. **Recipe: head-only FT** (freeze CLIP+DINO finer-32×32 backbone, train verb_proj+out) on human GT. ~50 labels = teacher parity, ~100 = clear win. Cheap: ~40s/epoch when box uncontended, 82k trainable params.
3. **Label budget: 100→200 objects**, prioritizing **grasp & display** (flat under FT, sparsely labeled) and novel/open-vocab verbs (lift/press/open, near-zero).
4. **Geometry: not on the critical path** for human-correctness. Keep only if a clean 32×32-DINO+geom test later shows a contain/sit gain worth the semantic-verb cost.
5. Engineering: cache features in RAM / decimate meshes so >100-object FT runs survive shared-box PCIe contention (the n172/full-FT runs were blocked purely by slow cold-cache feature reads, not by the method).

## ★ 5-FOLD CV — the headline, made rigorous 06-28 18:43
Single 172/58 split was NOT significant: gap +0.020, bootstrap 95% CI [-0.041,+0.077], P(FT>GEAL)=0.75 (small val: n=7 sit/move). Redid as 5-fold CV over all 225 labeled objects (each scored by the fold that held it out, ~180 train/fold, fixed 40ep, eval last.pt — no early-stop peeking):
  verb        n     FT    GEAL    gap
  contain   157  0.748  0.488  +0.260
  sit        29  0.817  0.534  +0.283
  pour       63  0.762  0.737  +0.025
  move       29  0.705  0.651  +0.055
  display    37  0.543  0.669  -0.126
  grasp      51  0.321  0.472  -0.151
  TRAINED-MEAN   0.650  0.592  +0.058   bootstrap 95% CI [+0.030,+0.085]  P(FT>GEAL)=1.00
VERDICT: human-FT beats GEAL SIGNIFICANTLY (+0.058) — but the win is concentrated in contain/sit (high-volume, well-rendered) and FT LOSES grasp/display (coverage-limited; GEAL's full-geometry PointNet++ wins there). The +0.058 reflects both lower noise AND more train data (~180/fold vs single-split 100) — both legitimate, consistent with the rising curve.
METHOD NOTE: single-split deltas at this scale (n=7-15/verb) are within noise — ALWAYS CV/bootstrap. We have 225 labeled objects (contain 157, pour 63, grasp 51, display 37, sit/move 29 each); reallocating to a bigger val isn't needed — k-fold uses every object as held-out once.
NEXT: Tier 2 (wider camera band, currently 25-60deg upper-hemisphere only -> ~36% verts invisible -> grasp 64% of missing positives are down-facing) + full retrain, targets grasp. Tier 3 geometry-fallback for contain/sit interior occlusions. Still UNBENCHMARKED vs recognized baselines (needed for field-level SOTA claim).

## TIER-2 (wider camera band) — PROTOTYPED then DROPPED 06-29 00:45
camera_sampling.py made env-overridable (AFFORD_ELEV_MIN/MAX_DEG; default 25/60 unchanged). Prototype: re-extracted wideband CLIP+DINO (rings -52,-33,33,52 = upper band + lower mirror, 32 views) for 6 grasp objects (bottle/cup/handbag).
RESULT: lower-hemisphere views recover only 20% of previously-invisible grasp positives (per-obj 1-33%; handbag handles best 30-33%, bottle bodies worst 1-10%); whole-object visible% rises just +2-8pp. The other 80% of invisible grasp surface is deep self-occlusion / single-image-unobserved (SAM3D hallucinates it) — no external camera reaches it.
KILL-SHOT: grasp's invisible positives were only ~14% of grasp positives anyway, and on the VISIBLE 86% the model already scores AUPRC 0.389 < GEAL 0.472. So grasp is APPEARANCE-hard even where seen; coverage was never the bottleneck. Even perfect coverage caps grasp ~0.39 << GEAL 0.47. Wider views CANNOT close the grasp/display gap.
VERDICT: Tier 2 not worth the heavy full-extraction+CV (days of gated GPU for a gain provably below GEAL). The grasp/display deficit is a FEATURE-TYPE problem (appearance vs geometry) -> GEAL's PointNet++ geometry wins there. Correct lever = TIER 3 (geometry as a complementary channel; mesh is complete regardless of visibility). Wideband prototype features left as vertex_semantics_wideband.pt/vertex_dino_wideband.pt on 6 objects only.

