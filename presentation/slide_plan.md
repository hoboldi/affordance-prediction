# 10-minute presentation — plan & assets

**Structure:** Introduction/Task/Background/Related Work · Method · Experiments · Conclusion + Next Steps.
**Framing:** near-final / progress talk — core result presented as achieved + validated; open items as a credible roadmap.

**Thesis (say it up front and again at the end):**
> From a single photo, we predict *where* on a 3D object you can grasp / pour / sit — verb-specific, with
> no manual labels — and the core challenge was making predictions actually *distinguish* between verbs,
> which a contrastive loss solved.

---

## 1. Introduction / Task / Background / Related Work  (~2 min — keep tight)
- **Task:** per-vertex, *verb-specific* affordance (grasp/pour/sit) on a 3D object reconstructed from a **single real photo**.
- **Why 3D, not 2D?** affordances are **geometric + actionable** — to act, an agent needs the **3D region on the surface**; a 2D heatmap is view-dependent and not directly plannable.
- **Why from an image?** 3D affordance data is **scarce** (scans/CAD); **images are abundant** → single-image 3D affordance is practical, "in the wild."
- **Related work (brief, name only):** 2D image affordances; 3D affordance on *clean* point clouds (**3D-AffordanceNet, OpenAD, GEAL, LASO**); single-image 3D (**SAM3D**) → **we bridge them: 3D affordance on single-image reconstructions, label-free.**
- *Figure:* teaser — `figures/showcase_v4_final.png` or `figures/04_prediction.png`.
> One-sentence motivation to say aloud: *"3D is the form you can act on, but 3D affordance data is expensive; images are cheap — so predicting 3D affordance from a single photo bridges that gap."*  Background tools (SAM3D/GEAL/CLIP) re-named in passing in Method M1; method/related details → Q&A. ~2 min.

## 2. Method  (~2.5 min — 2 slides)

### M1 — Pipeline, label-free  (~1.5 min)
- Single real photo → **SAM3D** → 3D mesh.
- Per-vertex features: **CLIP** (appearance, projected from multi-view renders) · **DINOv2** (object context) · **SLAT** (SAM3D geometry) · **normals**.
- → **verb-conditioned head** → per-vertex affordance map.
- Supervision: a **frozen GEAL teacher** run on the *same* mesh → pseudo-labels already in mesh-vertex order.
- ⇒ **no manual 3D labels, no cross-modal alignment** (the scalability hook — land this bullet).
- *Figure:* `01_input_photo → 02_reconstruction → 03_geal_label → 04_prediction` (verb = contain → bowl cavity).

### M2 — Conditioning + keeping verbs distinct  (~1 min)
- Verb vector **concatenated onto every vertex** → head predicts from (vertex, verb) jointly → can **localize a different region per verb** (a global modulation could only shift the whole field).
- Verb = **frozen CLIP *text* embedding** + small projection → **open-vocabulary** (any verb phrase, not a fixed table).
- Train: per-verb **BCE** on GEAL labels (balanced vertex sampling).
- ⚠ **BCE is computed per verb *independently*** → never compares verbs → head **collapses to a verb-agnostic "salient" average** (measured cross-verb corr **0.74**, vs teacher's **−0.14**).
- **Contrastive verb loss (hinged):** penalize cross-verb prediction correlation (clamp at 0) → push verbs to **disjoint** regions, *toward* the teacher's labels; hinge stops at disjoint (no over-separation).
- *Figure:* head diagram (inputs → concat → LayerNorm → MLP → logit), or before/after distinctness thumbnail.

## 3. Experiments  (~2.5 min — 2 slides: E1 setup+works+generalization, E2 distinctness→fix climax)
> Method + Experiments together ≈ **5 min** (the technical core).
- **Setup:** CO3D real photos, 10 categories, **vase/laptop held out**; metrics = AUPRC, cross-verb correlation, top-region IoU, prediction std.
- **It works:** qualitative across categories + unseen-category generalization. *Figure:* `figures/showcase_v4_final.png`.
- **The distinctness problem → diagnosis → fix (climax):** predictions collapsed to a verb-agnostic average (corr 0.74); the teacher's labels are **disjoint** (IoU 0) ⇒ it's the head, not the teacher; **contrastive loss → distinct + accurate** (corr **0.74 → −0.26**, IoU **0.56 → 0.00**, AUPRC preserved). *Figure:* **`figures/gt_comparison_contrastive.png`** (GEAL | before | after).
- **Generalization:** held-out vase/laptop AUPRC ~0.255 vs 0.274 seen.
- **Open-vocab:** any verb phrase; synonyms map correctly; characterized structure-vs-distinctness trade-off. *Figure:* `figures/showcase_v6_openvocab_contrastive.png`.

## 4. Conclusion + Next Steps  (~2 min)
- **Conclusion:** a *label-free*, verb-conditioned 3D-affordance pipeline that works on real images; the core distinctness challenge is **solved** (contrastive); open-vocab works with a characterized trade-off.
- **Next steps (roadmap):**
  1. **Honest evaluation** — all metrics are vs. the GEAL teacher; build a small **human-labeled** ground-truth set. *(Own this proactively.)*
  2. **Feature-extraction quality** — the per-vertex features show a **6-view seam artifact**; fix with **more views + multiple elevations + tighter framing**.
  3. **Fine-detail verbs** (press/open/lift) — teacher-capped; need a **2D affordance teacher**.
  4. **Structured + distinct open-vocab** — needs similarity scoring + feature↔text alignment (literature-aligned).

---

## Cut for time (mention only if asked)
- Model zoo (v2–v9, λ-sweeps, deeper proj, cross-attn) → "we iterated; the winning recipe was a hinged contrastive."
- Implementation details (LayerNorm, balanced sampling, checkpoints).

## Key numbers (use sparingly)
- Verb distinctness: cross-verb correlation **0.74 → −0.26**; top-region IoU **0.56 → 0.00** (matches teacher's disjoint labels); per-verb accuracy preserved.
- Generalization (held-out vase/laptop): AUPRC ~0.255 vs 0.274 seen.

## Assets (`figures/`)
`01_input_photo` · `02_reconstruction` · `03_geal_label` · `04_prediction` (pipeline) · `gt_comparison_contrastive.png` (before/after climax) · `showcase_v4_final.png` · `showcase_v6_openvocab_contrastive.png`. Full results table + narrative in `results_summary.md`.
