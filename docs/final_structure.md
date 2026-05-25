open_vocab_affordance/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/
│   ├── default.yaml
│   ├── model/
│   │   ├── sam3d.yaml
│   │   ├── vlm.yaml
│   │   ├── fusion.yaml
│   │   ├── mlp_head.yaml
│   │   └── transformer_head.yaml
│   │
│   ├── training/
│   │   ├── baseline.yaml
│   │   ├── multiview.yaml
│   │   └── visibility_aware.yaml
│   │
│   └── dataset/
│       ├── affordsplat.yaml
│       └── custom.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   ├── meshes/
│   ├── splats/
│   ├── renders/
│   └── annotations/
│
├── scripts/
│   ├── preprocess_affordsplat.py
│   ├── generate_sam3d.py
│   ├── render_gaussian_views.py
│   ├── render_views.py
│   ├── extract_vlm_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── visualize_affordances.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── affordsplat_dataset.py
│   │   ├── transforms.py
│   │   ├── mesh_loading.py
│   │   ├── view_sampling.py
│   │   └── collate.py
│   │
│   ├── reconstruction/
│   │   ├── __init__.py
│   │   ├── sam3d_wrapper.py
│   │   ├── mesh_utils.py
│   │   ├── gsplat_to_sam3d.py
│   │   ├── splat_utils.py
│   │   ├── pointnet_encoder.py
│   │   └── geometry_features.py
│   │
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── renderer.py
│   │   ├── gaussian_renderer.py
│   │   ├── mesh_renderer.py
│   │   ├── gaussian_ply.py
│   │   ├── gaussian_point_renderer.py
│   │   ├── camera_sampling.py
│   │   ├── visibility.py
│   │   └── rasterization.py
│   │
│   ├── vlm/
│   │   ├── __init__.py
│   │   ├── vlm_wrapper.py
│   │   ├── patch_extractor.py
│   │   ├── text_encoder.py
│   │   ├── intermediate_layers.py
│   │   └── feature_cache.py
│   │
│   ├── projection/
│   │   ├── __init__.py
│   │   ├── project_to_mesh.py
│   │   ├── barycentric.py
│   │   ├── patch_vertex_mapping.py
│   │   ├── multiview_fusion.py
│   │   └── visibility_weighting.py
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── concat_fusion.py
│   │   ├── attention_fusion.py
│   │   ├── cross_attention.py
│   │   └── feature_normalization.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   │
│   │   ├── heads/
│   │   │   ├── mlp_head.py
│   │   │   ├── transformer_head.py
│   │   │   ├── graph_head.py
│   │   │   └── uncertainty_head.py
│   │   │
│   │   ├── affordance_model.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── regularization.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── engine.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── checkpointing.py
│   │   └── mixed_precision.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_vertex.py
│   │   ├── evaluate_mesh.py
│   │   ├── evaluate_open_vocab.py
│   │   ├── visualization_metrics.py
│   │   └── benchmark.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── mesh_visualizer.py
│   │   ├── splat_visualizer.py
│   │   ├── heatmaps.py
│   │   ├── render_video.py
│   │   └── interactive_viewer.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── logging.py
│   │   ├── distributed.py
│   │   ├── reproducibility.py
│   │   ├── geometry.py
│   │   ├── device.py
│   │   └── config.py
│   │
│   └── experiments/
│       ├── baseline/
│       ├── intermediate_features/
│       ├── attention_fusion/
│       ├── visibility_aware/
│       └── transformer_head/
│
├── notebooks/                    # required: one notebook per pipeline stage
│   ├── 00_mesh_bootstrap.ipynb
│   ├── 01_affordsplat_dataloader.ipynb
│   ├── 01_reconstruction_debug.ipynb
│   ├── 02_rendering_mesh_debug.ipynb
│   ├── 03_rendering_gaussian_splat.ipynb
│   ├── 04_vlm_features_debug.ipynb
│   ├── 05_projection_debug.ipynb
│   ├── 06_affordance_head_debug.ipynb
│   ├── 07_training_evaluation_debug.ipynb
│   ├── 08_ablation_analysis.ipynb   # Phase 2+ comparisons
│   └── 10_sam3d_from_gsplat.ipynb
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── tensorboard/
│   ├── visualizations/
│   └── predictions/
│
├── tests/
│   ├── test_projection.py
│   ├── test_rendering.py
│   ├── test_vlm_features.py
│   ├── test_fusion.py
│   ├── test_losses.py
│   └── test_dataset.py
│
└── docs/
    ├── architecture.md
    ├── training.md
    ├── datasets.md
    ├── evaluation.md
    └── experiments.md
```

---

## Rendering backends and assets

| File | Role |
|------|------|
| `mesh.glb` | Required — affordance field, projection target, mesh renderer input |
| `gaussian.ply` | Optional — splat renderer input when SAM3D (or other) produces it |

`src/rendering/renderer.py` renders mesh depth + vertex correspondences, and replaces **RGB** from a 3DGS `.ply` when `rendering.backend` is `gaussian` or `both` and `splat_path` is set (gsplat with CUDA, else pyrender preview). Standalone `.ply` renders: `scripts/render_gaussian_views.py` / `gaussian_point_renderer.py`.

**Mesh-only development:** `data/sample.glb` is sufficient; set `rendering.backend: mesh` or `rendering.splat_path: null` for mesh-colour RGB without a paired splat.

**Full SAM3D output per sample:**

```text
outputs/reconstructions/{stem}/
    mesh.glb
    gaussian.ply      # optional for rendering; not needed for vertex affordances
    shape_latent.pt
    ...
```

---

## Stage testing notebooks

Notebooks are **required deliverables**, not optional demos. Each implements the same contract:

1. Prerequisites and config at the top.
2. Call into `src/` modules (not duplicated logic).
3. Inline visuals + numeric checks.
4. Pass checklist before the stage is considered done.

Caches and figures: `outputs/notebooks/<stage>/`. Details: [implementation_order.md](implementation_order.md#stage-testing-notebooks-required).