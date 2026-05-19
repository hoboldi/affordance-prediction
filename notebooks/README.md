# Stage testing notebooks

Full requirements: [docs/implementation_order.md](../docs/implementation_order.md#stage-testing-notebooks-required).

**Legend:** ✅ done · 🟡 partial · ⬜ not started · 🚫 blocked

| Notebook | Stage | Code ready | Status | Work on now? |
|----------|--------|------------|--------|--------------|
| `00_mesh_bootstrap.ipynb` | Mesh load / normalize | `datasets/mesh_loading.py` | ✅ | Run to verify locally |
| `01_reconstruction_debug.ipynb` | SAM3D | `reconstruction/sam3d_wrapper.py` | 🟡 | **No** — SAM3D checkpoints |
| `02_rendering_debug.ipynb` | Novel views (mesh) | `rendering/mesh_renderer.py` | ✅ | Run to verify locally |
| `03_vlm_features_debug.ipynb` | VLM patches | `vlm/` (CLIP) | ✅ | Run locally (downloads CLIP weights on first run) |
| `04_projection_debug.ipynb` | 2D→3D | `projection/project_to_mesh.py` | ✅ | Run after 02+03 caches |
| `05_affordance_head_debug.ipynb` | MLP head | `models/` stubs | ⬜ | **Yes** — uses `04_projection/vertex_semantic.pt` |
| `06_training_evaluation_debug.ipynb` | AGD20K eval | not started | ⬜ | AGD20K + full pipeline |
| `07_ablation_analysis.ipynb` | Ablations | — | ⬜ | Phase 2+ |

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/00_mesh_bootstrap.ipynb
jupyter lab notebooks/02_rendering_debug.ipynb
```

Caches: `outputs/notebooks/<stage>/`.
