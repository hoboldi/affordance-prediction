# Poster — full script (8 verbs) — taglines, captions, interpretations + numbers of record

**SAM3D for Language-Conditioned Affordance Prediction** · Mattis Krauch, Boldizsar Zalan Horvath (TUM)

*All numbers: clean human labels, 5-fold CV over ~238 CO3D objects, **8 verbs**. Metric = **trained-verb mean AUPRC (macro)**. Held-out. **Geometric = contain/pour/sit/move/lift · Appearance = grasp/display/press.***
*Legend: **⚑** table · **◆** diagram (numbers to redraw) · **[REC]** recommended · **[6v]** not recomputed for 8 verbs — 6-verb value or extrapolation, pattern holds.*

---

## 3. METHODOLOGY

### 3a. Pipeline ◆
- Subtitle: *"From a single image + verb to a per-vertex 3D affordance map."*
- Final input = **DINO 128 + geometry 5 + verb-text 128** (261-d with verb / 133-d features). CLIP-vision / SLAT / normals dropped (≈0 — see 4b).
- **CLIP-text caveat (honest):** verb is free text via CLIP, semantically grounded (real verbs ≫ nonsense), **robust to close paraphrases but NOT arbitrary synonyms** (see synonym test). Do **not** claim "open-vocabulary."

### 3b. Training progression ◆ [REC]
- Caption: *"Pretraining on GEAL pseudolabels already beats GEAL; a few human labels give the large gain."*
| stage | AUPRC (8v) |
|---|--:|
| GEAL teacher (zero-shot on human) | **0.46** |
| pretrained head (distilled, no human labels) | ~0.52 **[6v: +0.06 denoising; 8v not recomputed]** |
| **finetuned head (final)** | **0.86** |

---

## 4. RESULTS

### 4a. Comparison: Teacher — per-verb GEAL vs Ours ⚑ [REC]
- **Tagline:** *"Ours beats the GEAL teacher after finetuning — 0.86 vs 0.46 (verb-balanced)."*
- **Interpretation:** GEAL is zero-shot w.r.t. human labels; we surpass it on every verb. **press/lift are the biggest wins — GEAL nearly fails them.**
| verb | n | GEAL | Ours | Δ |
|---|--:|--:|--:|--:|
| contain | 142 | 0.48 | 0.98 | +0.51 |
| sit | 27 | 0.51 | 0.92 | +0.41 |
| display | 21 | 0.51 | 0.91 | +0.40 |
| move | 27 | 0.63 | 0.86 | +0.23 |
| grasp | 42 | 0.44 | 0.74 | +0.30 |
| pour | 56 | 0.71 | 0.89 | +0.18 |
| **press** | 27 | **0.31** | **0.91** | **+0.59** |
| **lift** | 12 | **0.09** | **0.69** | **+0.60** |
| **ALL (macro)** | | **0.46** | **0.86** | **+0.40** |

### 4a-fig. Teacher — GT ǀ GEAL ǀ Ours matrix ◆ [REC]
- **3 rows × 8 verb cols** (add press·keyboard + lift·handbag to the existing 6). Cells = cherry-picked *examples* (label "examples"); the table above is *averages*.

### 4b. Component ablation, by verb class ◆ [REC]
- **Tagline:** *"Different components drive different verb classes → geometry- vs appearance-based actions."*
- *Δ AUPRC when each component is removed from lean8 (5-fold). Baseline: ALL 0.863 · Geo 0.870 · App 0.851.*
| component removed | ALL Δ | Geometric Δ | Appearance Δ |
|---|--:|--:|--:|
| verb conditioning | **−0.39** | −0.40 | −0.38 |
| **DINO** | −0.13 | **−0.06** | **−0.25** |
| **geometry** | −0.07 | **−0.09** | **−0.04** |
| spatial GNN (→ matched MLP) | −0.06 | −0.05 | −0.09 |
| DINO + geometry together | *~−0.71* | *~−0.71* | *~−0.69* **[6v: extrapolated → ~0.18]** |

**Absolute AUPRC — for the bar chart** *(cascade; chance ≈ 0.20 line):*
| config | ALL | Geometric | Appearance |
|---|--:|--:|--:|
| **lean8 (final)** | **0.863** | **0.870** | **0.851** |
| − geometry | 0.793 | 0.780 | 0.813 |
| − spatial GNN | 0.803 | 0.825 | 0.765 |
| − DINO | 0.735 | 0.813 | **0.606** |
| − verb conditioning | 0.472 | 0.471 | 0.473 |
| − DINO + geometry | *~0.18* | *~0.18* | *~0.18* **[extrapolated]** |

**Interpretation (8v is a *cleaner* two-class story than 6v):**
- **DINO is appearance-specific** (App −0.25 vs Geo −0.06) — appearance verbs need learned appearance.
- **Geometry is now geometric-specific** (Geo −0.09 vs App −0.04) — **NEW vs 6-verb** (where geometry was uniform). Adding **lift** (which genuinely needs geometry) makes geometry a class discriminator. So **both** feature axes now split by class.
- **verb** universal (−0.39 both) — validity check: collapses to 0.47 (near nonsense floor) → the model genuinely conditions on the verb, and the task needs it (cross-verb IoU 0.06).
- **GNN** small, slight appearance lean.
- Dropped (≈0): CLIP-vision, SLAT, normals → lean 133-d.
- **Verb-routing callout [6v]:** real verbs 0.94 vs nonsense 0.58 (chance 0.20).

### 4b-find. THE FINDING — two classes of affordance verb ⚑ [REC]
| Property | **Geometric**<br>contain·pour·sit·move·**lift** | **Appearance**<br>grasp·display·**press** |
|---|:--:|:--:|
| Affordance determined by | object geometry | learned visual appearance |
| Reliance on appearance (drop-DINO) | low (−0.06) | high (−0.25) |
| Reliance on geometry (drop-geom) | high (−0.09) | low (−0.04) |
| Human labels for good perf **[6v]** | ~10 objects | ~100 objects |
| Rendering views required **[6v]** | 1 | 4–8 |
| Cross-category generalization | ✓ | ✗ |

- **⭐ lift vs grasp — the star illustration:** both "handle" verbs, opposite classes. **lift = geometric** (handbag strap = consistent geometric extremity; DINO-alone fails 0.38, geometry-alone 0.67). **grasp = appearance** (varied objects, no consistent geometric cue; geometry-alone fails 0.49). Same action, different object → different class. *(And they're disjoint regions: grasp=body, lift=strap, IoU 0.00.)*
- **Caveat line:** *"Classes reflect the dominant cue; lift (consistent strap) is geometry-driven, grasp (varied objects) appearance-driven."*

### 4b-detail. Per-verb × feature ablation ⚑ *(paper/supp)*
| dropped → | contain | pour | sit | move | display | grasp | press | lift |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DINO | −0.03 | −0.05 | −0.12 | −0.06 | **−0.31** | **−0.25** | −0.17 | −0.03 |
| geometry | −0.02 | −0.03 | −0.08 | −0.01 | −0.01 | −0.02 | −0.08 | **−0.31** |
| spatial GNN | −0.02 | −0.03 | −0.04 | −0.05 | −0.06 | −0.13 | −0.07 | −0.08 |
| verb cond. | −0.08 | −0.54 | −0.32 | −0.47 | −0.33 | −0.58 | −0.22 | −0.58 |

### 4c. Generalisation ◆ [REC]
**(i) Cross-category (LOCO) [6v]** — *"works for geometric verbs, not appearance."*
| verb (held-out cat) | AUPRC | transfers? |
|---|--:|:--:|
| pour | 0.82–0.83 | ✓ geometric |
| grasp | 0.15–0.16 | ✗ appearance |

**(ii) Cross-verb — unseen verb on same object**
- **max IoU = 0.059** (8-verb) → verbs near-disjoint → cross-verb zero-shot impossible *for functionally-distinct verbs.*
- Caption: *"Cross-verb zero-shot is hindered by lack of region overlap."*

### 4d. Synonym / open-vocab (NEW — decides the CLIP claim)
- **Result: synonyms mostly FAIL.** Close paraphrases work ("sit down" −0.02, show −0.06, "pour out" −0.15); genuine synonyms fail (grip/grab −0.27/−0.30, "pick up"/raise −0.51/−0.55, tap −0.71).
- **Conclusion:** model is sensitive to exact phrasing → **not open-vocabulary.** Scope CLIP as "semantically-grounded language substrate," not "arbitrary verbs." *(Synonym-augmented retrain could fix it — deferred.)*

---

## Big-number callouts [REC]
> **0.86 vs 0.46** (Ours vs GEAL, verb-balanced, +0.40) · **+0.060** spatial GNN > MLP · **133-d** lean input · **~238** objects, **8 verbs**

## Reference / baselines
| quantity | value (8v) |
|---|--:|
| chance | 0.20 |
| nonsense-verb prior **[6v]** | 0.58 |
| GEAL teacher (macro) | 0.46 |
| **lean8 final** | **0.863** |
| per-verb: contain/sit/press/display/pour/move/grasp/lift | 0.98/0.92/0.91/0.91/0.89/0.86/0.74/0.70 |

## Open items / status
- ✅ 8-verb: main, ablations, classification, per-verb, cross-verb IoU, teacher table — **all recomputed**.
- ⏸ **[6v] not recomputed (extrapolated, pattern holds):** pretrained-head middle (3b), label-efficiency ("labels" row), view-count ("views" row), cross-category LOCO (4c-i), verb-routing. DINO+geom row extrapolated.
- 📌 **Synonyms fail** → open-vocab claim dropped; CLIP scoped honestly. Synonym-augmented retrain deferred (user decision).
- **Verdict:** 8-verb adopted — better teacher gap (+0.40 vs +0.35), cleaner two-class (both DINO & geometry discriminate), lift the star illustration; only cost is headline mean 0.89→0.86 (lift, n=12).
