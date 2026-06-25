# Verb-conditioned affordance head — results summary (2026-06-24)

All numbers are **per-verb AUPRC** from `eval_verb_conditioning.py` (~50 multi-verb objects), which is
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
