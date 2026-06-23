# 10-minute presentation — plan & assets

**Thesis (say it slide 1 and again at the end):**
> From a single photo, we predict *where* on a 3D object you can grasp / pour / sit — verb-specific,
> with no manual labels — and the core challenge was making predictions actually *distinguish* between
> verbs, which a contrastive loss solved.

This is a **near-final / progress** talk: present the core result as achieved + validated, and the
remaining work as a credible, organized roadmap (not loose ends).

## Slide-by-slide (~10 slides, ~1 min each)

| # | Slide | Figure (in `figures/`) | ~Time |
|---|-------|------------------------|-------|
| 1 | Title + hook (thesis sentence) | `pipeline_figure.png` (panel 4) or `showcase_v4_final.png` | 0:30 |
| 2 | Problem & motivation — affordances are verb-specific + spatial; single image; 3D labels expensive | — | 1:00 |
| 3 | The idea (pipeline) — photo → SAM3D → features → verb-cond. head ← frozen-GEAL distillation, no manual labels | **`01_input_photo` → `02_reconstruction` → `03_geal_label` → `04_prediction`** (place as your own flow; verb = contain → bowl cavity) | 1:30 |
| 4 | Data — CO3D real photos, 10 categories, held-out split | — | 0:30 |
| 5 | It works — verb-conditioned affordances; generalizes to unseen categories | `showcase_v4_final.png` / `full_panel.png` | 1:00 |
| 6 | The problem we hit — verbs barely differ (cross-verb corr 0.74) | `gt_comparison_contrastive.png` (middle column) | 1:00 |
| 7 | Diagnosis (the insight) — labels are disjoint (IoU 0) but preds collapsed → it's the head, not the teacher | — | 1:00 |
| 8 | **The fix (climax)** — contrastive verb loss → distinct + accurate (corr 0.74→−0.26) | **`gt_comparison_contrastive.png`** (GEAL \| before \| after) | 1:30 |
| 9 | Open-vocabulary — any verb phrase via CLIP text; synonyms map correctly; characterized trade-off | `showcase_v6_openvocab_contrastive.png` | 1:00 |
| 10 | Status & roadmap (near-final) + thesis restated | — | 1:00 |

## Slide 3 — pipeline (draft)
**Title:** From one photo to verb-specific 3D affordances — without manual labels
Place the four standalone images as your own left-to-right flow: `01_input_photo` → `02_reconstruction` → `03_geal_label` → `04_prediction` (verb = contain → bowl cavity; rim & handles stay low).
Caption: *GEAL labels the **same** reconstructed geometry, so labels are already in mesh-vertex order — no cross-modal alignment, no hand annotation.*
Speaker notes (~75s): photo → SAM3D mesh → per-vertex CLIP/DINO/SLAT features → verb-conditioned head (verb via CLIP text = open-vocab) → per-vertex affordance map. Supervision: frozen GEAL on the same mesh → pseudo-labels → train with zero manual 3D annotation.

## Slide 10 — status & roadmap (near-final framing)
- **Where we are:** pipeline works end-to-end; the central challenge (verb-distinctness) is **solved** (contrastive result).
- **Remaining to final** (present as a plan, not loose ends):
  1. **Honest evaluation** — all metrics are vs. the GEAL teacher; next is a small human-labeled set for ground truth. *(Own this proactively — it's the obvious question.)*
  2. **Fine-detail verbs** (press/open/lift) — teacher-capped; needs a 2D affordance teacher.
  3. **Open-vocab structured+distinct** — characterized trade-off; literature-aligned fix (similarity scoring + feature-text alignment) is scoped.

## Cut for time (mention only if asked)
- The model zoo (v2–v9, λ-sweeps, deeper proj, cross-attn) → "we iterated; the winning recipe was a hinged contrastive."
- Implementation details (LayerNorm, balanced sampling, checkpoints).

## Key numbers (use sparingly)
- Verb distinctness: cross-verb correlation **0.74 → −0.26**; top-region overlap (IoU) **0.56 → 0.00** (matching the teacher's disjoint labels); per-verb accuracy preserved/improved.
- Generalization (held-out vase/laptop): AUPRC ~0.255 vs 0.274 seen.

(Full results table + narrative in `results_summary.md`.)
