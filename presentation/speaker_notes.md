# Speaker notes — keywords (glance, don't read aloud)

## Teacher–student / pseudo-labels  (Method M1)
- frozen **GEAL teacher** = pretrained 3D affordance model → run it on **our reconstructed mesh**
- output = **per-vertex pseudo-labels**, one heatmap per verb
- **why this:** manual 3D affordance labels are **expensive + scarce**
- teacher labels the **same geometry** we predict on → labels already in **vertex order**
  → **no cross-modal alignment, no label transfer**
- ⇒ **supervision for free**, every object auto-labeled → **scalable**
- student = our **lightweight verb-conditioned head**, learns by **distillation** from the teacher
- one-liner to say: *"no human labels — a frozen teacher labels the same mesh we predict on"*
- caveat to own (before they ask): metrics are **vs. the teacher**, not human ground truth → future work

## Affordance head / verb conditioning  (Method M2)
- per-vertex features: **CLIP · DINOv2 · SLAT · normals**
- **verb conditioning = CONCATENATION:** verb vector **appended to every vertex**
  → head predicts from **(vertex features, verb) jointly**
  → can **localize a different region per verb** (handle vs rim vs interior)
- contrast: a *global* modulation could only shift the whole field — concat **selects regions**
- say it, don't draw it: *"we concatenate the verb onto every vertex, so the same head lights up a different part per verb"*
- (Q&A only) LayerNorm on the input → keeps the verb vector from being drowned by unit-norm CLIP features
- head is **closed-set** here (learned verb embedding) — open-vocab text variant = future work

## Experiments
- setup: **CO3D · 9 categories · 6 verbs · train 987 / val 240**, evaluated **vs GEAL teacher**
- metrics: **AUPRC** = accuracy vs teacher · **std** = structured vs flat (flat ≈ random)
- **HEADLINE = the ablation (`ablation.png`):** feature quality drives accuracy
  - AUPRC: CLIP 6 views **0.22** → CLIP 16 views **0.26** → **+ per-vertex DINOv2 (16 views) 0.56** (the jump, base_l0)
  - say: *"per-vertex DINOv2 features drive the whole gain"*
  - (if asked) a contrastive loss on top changed nothing → we don't use it
- **teacher fidelity** (`teacher_student`): predictions track the GEAL teacher across categories
- **verb-specificity** (`specificity_cup`): grasp vs contain **anti-localized** (handle vs body) → not generic saliency
- **generalization to UNSEEN categories** (`generalization.png`): vase + laptop were **fully held out** (the val split is by category, not instance) — model still localizes: vase pour→rim, contain→body, laptop display→screen
- **limitations (own them):** predictions **broader/softer than teacher** (handbag-grasp = whole bag vs strap); fine verbs (display/sit) weaker; **tv only 27 samples**; all metrics **vs teacher, not human GT**
- **open-vocab WORKS** (`openvocab.png`, v15_ov_nc — CLIP-text verb, **no contrastive loss**): AUPRC ~0.44, std ~0.13 (structured)
  - **unseen synonyms localize to the right region:** grip≈grasp→handle · sip≈pour→rim · fill≈contain→interior
  - **key finding: the contrastive loss was *flattening* open-vocab** — the ov runs that used it (v9, v14) collapsed (std ~0.02); dropping it (v15) fixed it
  - caveat: v15 still training (cite as current, epoch ~20); contain/fill softer than grasp/pour
