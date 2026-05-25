# Pipeline: gsplat views → SAM3D

This connects **dataset 3D Gaussians** (e.g. AffordSplat `Gaussian/GS_*.ply`) to the repo’s **SAM3D** wrapper:

1. **Rasterise** the splat with **true `gsplat`** (`render_gaussian_splat_gsplat_views`) using the same orbit cameras as mesh rendering (`configs/default.yaml` → `rendering`).
2. Write **`images/`** + **`masks/`** in the layout expected by `scripts/generate_sam3d.py` (full-foreground masks; SAM3D still uses **one** RGB for inference).
3. Call **`SAM3DWrapper.reconstruct`** on `view_{REFERENCE:03d}.png` (default `view_000.png`).
4. Persist **`mesh.glb`**, **`gaussian.ply`**, latents, and `meta.json` under the run’s `reconstruction/` directory; optionally cache **`global_latent.pt`** under `paths.cache_root/sam3d/<stem>/` when `reconstruction.cache_latents` is true.

## Code entry points

| Location | Role |
|----------|------|
| [`src/reconstruction/gsplat_to_sam3d.py`](../src/reconstruction/gsplat_to_sam3d.py) | `export_gsplat_views_for_sam3d`, `run_sam3d_on_prerendered_view`, `gsplat_ply_to_sam3d_reconstruction` |
| [`scripts/render_gsplat_and_sam3d.py`](../scripts/render_gsplat_and_sam3d.py) | CLI: `--splat_path`, `--run_dir`, `--reference_view`, … |
| [`notebooks/10_sam3d_from_gsplat.ipynb`](../notebooks/10_sam3d_from_gsplat.ipynb) | Interactive run + error hints |

## Requirements

- **CUDA** + **`gsplat`** for step 1.
- **SAM3D** code + weights: the upstream repo must contain **`notebook/inference.py`**. Clone with `git submodule update --init sam-3d-objects`, set **`SAM3D_OBJECTS_ROOT`** if the checkout lives elsewhere, or use the Docker image (bundled under `/workspace/sam-3d-objects` — do not replace that path with an empty bind-mounted folder). **Weights** are not in git: request access on [Hugging Face — facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects), then follow **`sam-3d-objects/doc/setup.md` §2** so `sam-3d-objects/checkpoints/hf/pipeline.yaml` exists (`reconstruction.sam3d_config`).

If you see **`ModuleNotFoundError: No module named 'inference'`**, that is this missing or hidden SAM3D tree — not a missing `pip install gsplat`.

If you see **`No such file or directory: …/checkpoints/hf/pipeline.yaml`**, run the HF download steps in that same doc section; `HF_TOKEN` / `hf auth login` is usually required.

## Limitations

- SAM3D’s public API here is **single-image**. Extra rendered views are for inspection or a future multi-view fusion path; they are not all fed into one SAM3D forward pass in this module.
