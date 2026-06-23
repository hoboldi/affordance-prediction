# Overnight results: verb-conditioning via contrastive loss

**Goal addressed:** "on most samples the affordance head is not distinct enough — only slight differences
between verbs." This run diagnosed why and fixed it.

## Diagnosis (why verbs weren't distinct)
The GEAL teacher labels are **near-disjoint across verbs** (cross-verb corr **−0.14**, top-10% region
IoU **0.000**), but the head's predictions **collapsed to a verb-agnostic average** (corr 0.74–0.84,
IoU 0.56). So it was the **head/loss, not the teacher** — per-verb BCE gives no pressure for cross-verb
distinctness, and the geometry features predict a generic "salient region" that earns partial credit on
every verb. (Mesh detail, point-sampling density, and unknown categories were all ruled out earlier.)

## Fix: contrastive verb loss (+ a hinge)
Added a per-object term penalizing cross-verb prediction correlation (`--contrastive_weight`).
- **λ-sweep gotcha:** a raw correlation penalty has *no floor* → it over-separates to corr ≈ **−0.7** at
  *every* λ (0.3/0.5/1.0), distorting common verbs (contain AUPRC crashed 0.25→0.06).
- **Hinge fix:** penalize only *positive* correlation (clamp at 0) → pushes verbs to uncorrelated/disjoint
  and then stops. This is the win.

## Results (final epoch, 60-object eval; "*" = held-out category)
Conditioning — cross-verb correlation (↓ = more distinct), top-region IoU, prediction std (contrast):

| model | conditioning | corr | IoU | std |
|-------|--------------|------|-----|-----|
| v3 | learned, no contrastive | 0.74 | 0.56 | 0.12 |
| v4 | open-vocab, no contrastive | 0.84 | 0.56 | 0.14 |
| v5 | learned, raw-corr λ=1 | −0.47 | 0.00 | 0.12 |
| v5b/c | learned, raw-corr λ=0.3/0.5 | −0.40 | 0.00 | 0.13 |
| **v5d** | **learned, HINGE λ=1** | **−0.26** | **0.00** | **0.07** |
| v6 | open-vocab, HINGE λ=1 | −0.37 | 0.00 | 0.009 |
| _GEAL labels_ | _reference_ | _−0.14_ | _0.00_ | — |

Per-verb AUPRC (vs GEAL):

| verb | v3 | v5 (raw) | **v5d (hinge)** | v6 |
|------|----|---------|-----------------|----|
| contain | 0.25 | 0.15 | **0.24** | 0.22 |
| grasp | 0.18 | 0.13 | **0.21** | 0.18 |
| pour | 0.18 | 0.09 | **0.20** | 0.16 |
| sit | 0.37 | 0.44 | **0.46** | 0.26 |
| move | 0.17 | 0.10 | 0.12 | 0.13 |

Per-category AUPRC + generalization (gen-eval):

| | v4 | **v5d** | v6 |
|--|----|---------|----|
| SEEN categories mean | 0.20 | **0.274** | 0.216 |
| HELD-OUT (vase, laptop) | 0.175 | **0.255** | 0.191 |

Open-vocab synonym-consistency — gap = corr(synonym, trained) − corr(unrelated, trained), higher better:

| model | synonym corr | unrelated corr | **GAP** |
|-------|-------------|----------------|---------|
| v4 (open-vocab) | +0.94 | +0.84 | +0.105 |
| **v6 (open-vocab + contrastive)** | +0.65 | −0.26 | **+0.910** |

## Conclusions
1. **v5d (learned + hinged contrastive) is the best model** — verbs are now *disjoint* (IoU 0, matching
   the labels) **and** AUPRC is preserved/improved on 4/5 verbs, with the best per-category +
   held-out generalization. This directly resolves the "not distinct enough" problem.
2. **Contrastive transforms open-vocab discrimination** — v6's synonym/unrelated gap jumped +0.105 → **+0.910**
   (synonyms now clearly behave like their verb, unrelated verbs anti-correlate). But combining open-vocab
   *and* contrastive flattened predictions (std 0.009), so it trades contrast for discrimination — it does
   not beat the parts. Best fixed-verb model = v5d; best structured open-vocab = v4.
3. **Cross-attention was unnecessary** — the limitation was the *loss*, not the conditioning mechanism;
   the hinged contrastive on the existing concat head fixed it.

## Figures
- `outputs/renders/gt_comparison_contrastive.png` — GEAL | v3 | v5d (the distinctness win).
- `outputs/renders/showcase_v6_openvocab_contrastive.png` — open-vocab triplets (trained/synonym/new-action).
- `outputs/renders/showcase_v4_final.png`, `outputs/renders/gt_comparison.png` — earlier (v3/v4).

## Caveats & next steps
- **All metrics are still vs GEAL** (a proxy). The biggest remaining gap is a **small human-labeled eval set**
  (the 2D-scribble→project tool) to get ground truth — without it we can't know absolute quality.
- **Fine verbs (press/open/lift)** remain teacher-capped (GEAL CAD→real domain gap) — needs a 2D
  affordance teacher, not addressed here.
- **Optional tuning:** v6's flatness might be recoverable with a smaller contrastive λ for the open-vocab head.
- **move** is the one verb contrastive didn't help (data-scarce: chairs only).
