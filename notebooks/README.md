# Stage testing notebooks

Full requirements: [docs/implementation_order.md](../docs/implementation_order.md#stage-testing-notebooks-required).

**Legend:** ✅ done · 🟡 partial · ⬜ not started · 🚫 blocked

| # | Notebook | Stage | Code ready | Status | Work on now? |
|---|----------|--------|------------|--------|--------------|
| 0 | `00_mesh_bootstrap.ipynb` | Mesh load / normalize | `datasets/mesh_loading.py` | ✅ | Run to verify locally |
| 1 | `01_affordsplat_dataloader.ipynb` | AffordSplat / 3DAffordSplat (local) | `datasets/affordsplat_local_dataset.py` | ✅ | **Yes** — auto root + `AffordSplatLocalDataset` |
| 2 | `02_rendering_mesh_debug.ipynb` | Novel views (mesh) | `rendering/mesh_renderer.py` | ✅ | Run to verify locally |
| 3 | `03_rendering_gaussian_splat.ipynb` | Novel views (3DGS / gsplat) | `gaussian_gsplat_renderer.py` | ✅ | PNG/JPEG → **`exports/gaussian_splat/`** (visible; `outputs/` is gitignored) |
| 4 | `04_vlm_features_debug.ipynb` | VLM patches | `vlm/` (CLIP) | ✅ | After **02** caches (or re-render inline) |
| 5 | `05_projection_debug.ipynb` | 2D→3D | `projection/project_to_mesh.py` | ✅ | After **02** + **04** caches |
| 6 | `06_affordance_head_debug.ipynb` | MLP head | `models/` stubs | ⬜ | **Yes** — uses `05_projection/vertex_semantic.pt` |
| 7 | `07_training_evaluation_debug.ipynb` | 3DAffordSplat train/eval | not started | ⬜ | AffordSplat data + full pipeline |
| 8 | `08_ablation_analysis.ipynb` | Ablations | — | ⬜ | Phase 2+ |
| 10 | `10_sam3d_from_gsplat.ipynb` | gsplat views → SAM3D | `reconstruction/gsplat_to_sam3d.py` | 🟡 | **GPU + gsplat + SAM3D weights**; see `scripts/render_gsplat_and_sam3d.py` |

**Note:** SAM3D reconstruction (`01_reconstruction_debug.ipynb`) is still planned once checkpoints are available. **`01_affordsplat_dataloader.ipynb`** covers the local Hugging Face mirror under `/data`.

**Typical run order (MVP):** `00` → **`01`** (optional) → `02` → `03` (optional gsplat) → **`10`** (optional: SAM3D from gsplat views, needs weights) → `04` → `05` → `06`.

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/00_mesh_bootstrap.ipynb
jupyter lab notebooks/02_rendering_mesh_debug.ipynb
```

Caches: `outputs/notebooks/<stage>/` (e.g. `02_rendering`, `04_vlm`, `05_projection`).
