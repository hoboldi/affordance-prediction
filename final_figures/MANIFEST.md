# Final figures — poster & paper

All figures for the final verb-conditioned 3D affordance model, one clean place.
**All re-rendered clean:** contaminated labels excluded (54 GEAL-pseudolabel files
quarantined + a `frac_intermediate < 0.005` binary-fraction filter), **no
top-of-figure text**, examples **cherry-picked for visible regions** (no
invisible-interior cases like contain-on-bottle).

**Provenance:** every prediction is **held-out** — each object is scored by the
fold model that never trained on it (`gnn_full_preclean_fold{0-4}`, the 0.895
full-feature GEAL-pretrain GNN). The lean final model (0.890) is within noise and
visually identical, so caption these as *the model*, not two different models.

| clean name | source (`outputs/renders/…` unless noted) | poster section | shows |
|---|---|---|---|
| `method_pipeline.png`         | *drawn — `scratchpad/render_pipeline.py`* | Methodology (inference) | banner: RGB image + verb-text → DINO / geometry / verb branches → GNN head → affordance map |
| `train_pipeline.png`          | *drawn — `scratchpad/render_train_pipeline.py`* | Methodology (training) | banner: distill frozen GEAL teacher → pretrain GNN → warm-start → fine-tune on 226 human labels (5-fold) → final model |
| `task_verb_routing.png`       | verb_routing_clean.png   | Task | SAME object, different verb → different region (chair: sit/move; cup: pour/grasp) |
| `task_affordance_wheel.png`   | *drawn — `scratchpad/render_wheel3.py` (+`wheel_lib.py`)* | Task (alt to verb_routing) | radial wheel: ONE mug (cup__12_100_593) → 3 mutually-distinct trained verbs as spokes, verb labels only (grasp→handle · pour→rim · contain→inside, regions NOT labeled). Center = grey geometry (no label). Sharp poster-style maps; verbs are near-independent (pairwise corr grasp·pour −0.05, grasp·contain −0.29, pour·contain +0.05). **Model: `geomclean_fold0` (FiLM verb-conditioning via dino_cls/ss_dino_cls + `vertex_dino_fine.pt`)** — NOT gnn_geom (which lacks dino_cls FiLM → verbs collapse). Regen: `PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python scratchpad/render_wheel3.py --obj cup__12_100_593__v000 --az 20 --el 26 --spec "grasp,contain,pour" --out final_figures/task_affordance_wheel.png` (append `\|region` per verb to re-enable region captions) |
| `result_geal_matrix.png`      | *drawn — held-out renders* | Results (**HERO**) | 3 rows (GT \| GEAL teacher \| Ours) × 6 verb cols (contain·bowl, sit·chair, pour·bottle, move·chair, display·tv, grasp·cup). GEAL misplaced/blobby/underfilled (0.12–0.49), Ours sharp+correct (0.93–1.00). Cell AUPRC are best-case cherry-picks — pair with `result_geal_table` for the honest 0.53→0.92 average. |
| `result_geal_table.png`       | *drawn — GEAL vs Ours*   | Results (with hero) | honest per-verb averages (held-out, n=315): GEAL 0.53 → Ours 0.92 overall (contain +0.51 … pour +0.19) |
| `result_gnn_vs_mlp.png`       | hero_gnn_vs_mlp_clean.png | Results (**→ architecture ablation slot**) | GT \| MLP \| GNN, 3 rows (grasp handbag 0.26→0.95, grasp cup 0.30→0.93, display laptop 0.63→1.00): MLP speckled/wrong, GNN clean |
| `result_gallery.png`          | gallery_clean.png        | Results | 2×4 grid of good GT↔GNN pairs, all 6 trained verbs, 7 categories (contain·bowl, sit·chair, pour·bottle, grasp·handbag, display·laptop, move·chair, grasp·cup, pour·vase) |
| `result_ablation.png`         | *drawn — ablation bars*  | Results | contribution of each **final-model** component (drop from lean 0.890, 5-fold): verb −0.34 ≫ DINO −0.14 ≫ spatial-GNN −0.05 ≈ geom −0.03 |
| `result_per_verb.png`         | *drawn — per-verb bars*  | Results | per-verb AUPRC (5-fold), colored by class: geometric ~0.87–0.98 uniform; appearance = higher variance (display 0.91, **grasp 0.77** = outlier-low) |
| `result_two_class.png`        | *drawn — 2-class table*  | Results (the finding) | **geometric vs appearance verbs**: needs-DINO ≈0/−0.33, labels ~10/100+, views 1/4–8, cross-category ✓/✗ |
| `gen_crossverb_iou.png`       | verb_overlap_heatmap.png | Generalization | cross-verb region IoU max 0.06 → zero-shot structurally impossible |
| `gen_crosscategory_loco.png`  | loco_generalization.png  | Generalization | leave-one-category-out: pour transfers (0.82/0.83), grasp fails (0.15/0.16) |
| ~~`gen_fewshot.png`~~ (DROPPED) | fewshot_progression.png | — | few-shot removed from poster: all novel verbs (press/lift/open) are homogeneous → 0.12→0.80 is favorable-case-only, not a general claim |
| `result_sample_efficiency.png`| label_efficiency.png     | Experiments (Analysis) | GEAL-pretrain vs scratch, AUPRC vs #labels: gap +0.25 at n=10 → ~0 at full (label efficiency; mostly on appearance verbs) |
| `result_view_count.png`       | view_count.png           | Experiments (Analysis) | per-verb vs #DINO views: geometric flat ~0.93, **grasp 0.62→0.77** (occlusion) |
| `karahi_gt_vs_pred.png`       | *drawn — `scripts/_render_karahi_demo.py`* | Demo / qualitative | in-the-wild **karahi** (single phone photo → SAM3D recon): human GT vs GNN prediction, 2 rows (contain·basin, grasp·handles) × GT\|Pred. Verb-conditioned routing on ONE unseen object. AUPRC vs human GT: contain 0.66, grasp 0.89. Individual tiles: `karahi_{contain,grasp}_{gt,pred}.png`. |

## Key numbers (for callouts)
- MLP 0.814 → **GNN 0.870** (+0.056, all folds) → **+GEAL-pretrain 0.895**; lean final 0.890.
- Features: DINO +0.057, geom +0.020, CLIP/SLAT/normals ≈0 → lean 133-d input.
- Verb routing: real 0.938 vs nonsense 0.584.
- **GEAL teacher (zero-shot on human) 0.53 → Ours 0.92 = +0.39** (held-out, n=315; the old "+0.058" was the head-only model, now corrected).
- Component ablation (drop from lean 0.890): verb −0.34 ≫ DINO −0.14 ≫ GNN −0.05 ≈ geom −0.03.
- Cross-verb region IoU max **0.06**. 226 human-labeled objects, 5-fold CV, AUPRC.

## Notes
- `method_pipeline.png` is a drawn schematic (not a model render). Source:
  `scratchpad/render_pipeline.py` — edit colors/wording and re-run to regenerate.
  "render 16 views" is on the DINO branch only; geometry needs no render; verb-text
  is a separate input.
- The others are re-rendered by `scratchpad/render_{hero,gallery,routing,overlap_heatmap,loco,fewshot_prog}.py`.
  Gallery picks come from `scratchpad/contact_sheet.png` (GT-colored grid, high-GNN + visible);
  the hero rows are the top GNN−MLP gaps from `scratchpad/gnn_wins_scan.py` (`gnn_wins.json` / `gnn_wins_sheet.png`).
- Still model renders (turbo = high affordance). Poster captions carry the message
  since the figures themselves are text-free.
