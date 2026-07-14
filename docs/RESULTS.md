# ReVerb — Results

All numbers are 5-fold cross-validation over **objects** (every object scored by the fold that never
trained on it during human finetuning), verb-balanced **macro AUPRC**, positive base rate ≈ 0.20.

## Dataset

238 objects reconstructed from [CO3D](https://github.com/facebookresearch/co3d) across 10 categories;
215 carry human verb–region annotations — **363 object–verb annotations** over 8 verbs (354 with a
non-empty positive region, which AUPRC can score).

| verb | annotations (scoreable) |
|---|---|
| contain | 142 |
| pour | 56 |
| grasp | 42 |
| sit | 27 |
| move | 27 |
| press | 27 |
| display | 21 |
| lift | 12 |

## Distillation transfer (GEAL teacher → ReVerb, zero-shot teacher on reconstructed meshes)

The distilled head exceeds its teacher before any human finetuning (macro 0.46 → 0.52, evaluated on
the distillation pool). This table quantifies pseudo-label transfer under domain shift — **not** a
benchmark against GEAL (GEAL consumes point clouds and was trained on PIAD-C/LASO-C).

| verb | GEAL | ReVerb | Δ |
|---|---|---|---|
| contain | 0.48 | 0.98 | +0.51 |
| sit | 0.51 | 0.92 | +0.41 |
| display | 0.51 | 0.91 | +0.40 |
| pour | 0.71 | 0.89 | +0.18 |
| move | 0.63 | 0.86 | +0.23 |
| grasp | 0.44 | 0.74 | +0.30 |
| press | 0.31 | 0.91 | +0.60 |
| lift | 0.09 | 0.69 | +0.60 |
| **ALL** | **0.46** | **0.86** | **+0.40** |

## Component ablation

| Configuration | All | Geometric | Appearance |
|---|---|---|---|
| ReVerb (full) | 0.863 | 0.870 | 0.851 |
| − spatial GNN (per-vertex MLP) | 0.803 | 0.825 | 0.765 |
| − geometry | 0.793 | 0.780 | 0.813 |
| − DINO | 0.735 | 0.813 | 0.606 |
| − verb conditioning | 0.472 | 0.471 | 0.473 |
| − DINO & geometry | 0.173 | 0.186 | 0.151 |

- **Verb conditioning is essential** — replacing the verb with a constant embedding collapses the
  model to 0.47 (it can no longer route affordance to verb-specific regions).
- **DINO + geometry jointly carry the per-vertex signal** — removing both drops to 0.173.
- The GNN over an MLP of comparable capacity is a modest number (0.863 vs 0.803) but a large
  qualitative gain: contiguous maps instead of speckled ones.

## The two-verb-class finding

Removing DINOv2 cleanly separates the verbs into two classes, while removing geometry does not:

| removed channel | Geometric Δ | Appearance Δ |
|---|---|---|
| DINOv2 | 0.045 [0.033, 0.058] | 0.240 [0.182, 0.296] |
| geometry | 0.041 [0.030, 0.052] | 0.040 [0.015, 0.065] |

Brackets are bootstrap 95% CIs over resampled object–verb items. The DINO split is a factor of five
with non-overlapping intervals; the geometry channel does not separate the classes (the apparent
effect is driven by *lift*, n=12). So the robust axis is **appearance-reliance**, not geometry.

The split survives every confound we tested:
- **Region size** (appearance regions are smaller): the gap persists on a size-matched subset
  (+0.122, 95% CI [+0.068, +0.183]) and under a regression control on positive-region fraction.
- **Single-verb dominance**: leaving out any one verb keeps the gap in [0.170, 0.223].
- **The teacher**: before human finetuning the distilled head barely uses DINOv2 (+0.064 appearance,
  −0.00 geometric), so the reliance is learned *from the affordance supervision*, not inherited.

(Text deltas pool object–verb items; the ablation table balances verbs — since the geometric class is
dominated by *contain*, item-pooling reports 0.045 where verb-balancing reports 0.057. The separation
holds either way.)

## Cross-category generalization

Holding out an entire object category at training time (models trained from scratch on the remaining
categories — no distillation, so the held-out category is unseen by any supervision):

- **Geometric verbs transfer**: with bottles held out, *pour* reaches 0.82.
- **Appearance verbs do not**: *grasp* collapses to 0.15 (0.15–0.51 across the three held-out
  categories), because it depends on category-specific visual cues.

## Cross-verb (open-vocabulary) scope

The verb input is real language (CLIP text), but bounded. Exact verbs and close paraphrases work;
arbitrary synonyms do not (held-out synonyms lose ≈0.30 macro AUPRC). The bottleneck is CLIP's
text-embedding geometry, which is tuned for image–text rather than text–text similarity — it places
2/8 held-out synonyms next to their target verb, versus 8/8 for a sentence-similarity encoder
(all-mpnet). A preliminary encoder swap helps ambiguous words without hurting exact-verb accuracy but
does not fully close the gap: the limitation is the text encoder, not the affordance head.

## Limitations

- **Supervision lives on the reconstructed mesh.** Pseudo-labels, human annotations, and scoring are
  all defined on the SAM3D reconstruction; we do not verify correspondence to the physical object. All
  numbers should be read in the reconstructed frame.
- **No valid benchmark against GEAL** (different input modality and training data; evaluated zero-shot).
- **Scale.** 215 objects / 363 annotations; per-verb tests are underpowered (most acutely *lift*, n=12),
  and true cross-verb generalization is not testable (annotated regions are near-disjoint).
