# Pipeline: gsplat views → SAM3D

> **You cannot use this outside the container** in a supported way. The gsplat → SAM3D path is pinned and tested only in the **`sam3d-pipeline` Docker** image (CUDA, `gsplat`, `flash-attn`, `sam-3d-objects`). Use `docker compose run --rm sam3d-pipeline bash` and run from `/workspace`; see [docker/README.md](../docker/README.md).

This connects **dataset 3D Gaussians** (e.g. AffordSplat `Gaussian/GS_*.ply`) to the repo’s **SAM3D** wrapper:

1. **Rasterise** the splat with **true `gsplat`** (`render_gaussian_splat_gsplat_views`) using the same orbit cameras as mesh rendering (`MeshRenderConfig` / `configs/default.yaml` → `rendering`).
2. Write **`images/`** + **`masks/`** in the layout expected by `scripts/generate_sam3d.py` (full-foreground masks; SAM3D still uses **one** RGB for inference).
3. Call **`SAM3DWrapper.reconstruct`** on `view_{REFERENCE:03d}.png` (default `view_000.png`).
4. Persist **`mesh.glb`**, **`gaussian.ply`**, SLAT tensors, **`global_latent.pt`** (mean-pooled SLAT for the whole object, same format as the cache file), **`slat_vertex_features.pt`**, and `meta.json` under the run’s **`reconstruction/`** directory. Optionally also cache **`global_latent.pt`** under `paths.cache_root/sam3d/<stem>/` when `reconstruction.cache_latents` is true (training shortcuts).

`gsplat_ply_to_sam3d_reconstruction` (and `scripts/generate_sam3d.py`) run SAM3D inside **`sam3d_environment()`** in [`sam3d_wrapper.py`](../src/reconstruction/sam3d_wrapper.py): it prepends the `sam-3d-objects` tree to `sys.path` for that block, then **removes those prefixes** and restores **`LIDRA_SKIP_INIT`** so later code (including `gsplat`) does not keep SAM3D’s skip flag. **`CUDA_HOME`** is still defaulted from `CONDA_PREFIX` when unset and is left in place afterward so CUDA tooling stays stable.

When **`orbit_ring_rotation_deg`** is non-zero on the render config used for prerender, **gsplat → SAM3D** applies the **inverse** of that same fixed-axis rotation to the decoded mesh (and to SAM3D Gaussian means + quaternions when `pytorch3d` is available) before writing files, so artifacts stay in the **same normalized splat frame** as AffordSplat / GT (cameras were tilted via the ring; splat means were not).

**gsplat colour looks grey with rare coloured specks?** Large PLYs are **subsampled** to `max_points` (default **500 000** in `gsplat_ply_to_sam3d_reconstruction` / CLIs; raise further if needed). Subsampling defaults to **voxel stratification** so every occupied region of the bbox contributes Gaussians (uniform random oversamples small high-contrast parts like knobs). Set `AFFORDANCE_GSPLAT_SUBSAMPLE=uniform` to revert. Optional: `AFFORDANCE_GSPLAT_F_REST_LAYOUT=rgb_interleaved` for non-Inria `f_rest_*` order; `AFFORDANCE_GSPLAT_SH_DEGREE_CAP=0` forces DC-only tint (see `rendering/gaussian_gsplat_renderer.py`).

| Location | Role |
|----------|------|
| [`src/reconstruction/gsplat_to_sam3d.py`](../src/reconstruction/gsplat_to_sam3d.py) | `export_gsplat_views_for_sam3d`, `run_sam3d_on_prerendered_view`, `gsplat_ply_to_sam3d_reconstruction` |
| [`src/reconstruction/gsplat_sam3d_batch.py`](../src/reconstruction/gsplat_sam3d_batch.py) | `ensure_sam3d_reconstruction_for_splat`, `batch_ensure_sam3d_reconstructions` — **idempotent** runs (skip if `mesh.glb` + matching `meta_prerender.json` exist) for pipelines / notebooks |
| [`scripts/render_gsplat_and_sam3d.py`](../scripts/render_gsplat_and_sam3d.py) | CLI: `--splat_path`, `--run_dir`, `--reference_view`, … |
| [`scripts/batch_gsplat_sam3d.py`](../scripts/batch_gsplat_sam3d.py) | CLI: many `.ply` paths, `--splat_paths_file`, or `--affordsplat_random_n` under one `--output_root` (default `exports/gsplat_sam3d_runs`) |
| [`notebooks/10_sam3d_from_gsplat.ipynb`](../notebooks/10_sam3d_from_gsplat.ipynb) | Interactive run + error hints |

## Requirements

- **CUDA** + **`gsplat`** for step 1.
- **SAM3D** code + weights: the upstream repo must contain **`notebook/inference.py`**. Clone with `git submodule update --init sam-3d-objects`, set **`SAM3D_OBJECTS_ROOT`** if the checkout lives elsewhere, or use the Docker image (bundled under `/workspace/sam-3d-objects` — do not replace that path with an empty bind-mounted folder). **Weights** are not in git: request access on [Hugging Face — facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects), then follow **`sam-3d-objects/doc/setup.md` §2** so `sam-3d-objects/checkpoints/hf/pipeline.yaml` exists (`reconstruction.sam3d_config`).

If you see **`ModuleNotFoundError: No module named 'inference'`**, that is this missing or hidden SAM3D tree — not a missing `pip install gsplat`.

If you see **`No such file or directory: …/checkpoints/hf/pipeline.yaml`**, run the HF download steps in that same doc section; `HF_TOKEN` / `hf auth login` is usually required.

## Downstream (rendering → VLM → projection)

After a successful run, **`RUN_DIR`** (notebook 10) is the value to pass everywhere as **`SAM3D_RUN_DIR`**: the folder that contains **`reconstruction/mesh.glb`**.

Use **`reconstruction.mesh_utils.cfg_with_sam3d_reconstruction(cfg, RUN_DIR)`** (or set the same `SAM3D_RUN_DIR` in notebooks **02**, **04**, **05**, **06**) so `rendering.mesh_path` points at that mesh (and `gaussian.ply` when present). Then **delete** `outputs/notebooks/02_rendering/` if you previously cached renders for another mesh, and rerun **02 → 04 → 05 → 06** so vertex counts match.

**Global latent in the affordance head:** each run writes **`reconstruction/global_latent.pt`** (same `{"global_latent": (8,)}` payload as the cache helper) next to **`slat_vertex_features.pt`**. With `reconstruction.cache_latents: true` (default), notebook 10 may **also** mirror that tensor under **`paths.cache_root/sam3d/<stem>/global_latent.pt`**. Notebook **06** loads it via **`reconstruction.sam3d_wrapper.try_load_cached_global_latent`** when **`SAM3D_RUN_DIR`** (and `reconstruction/meta.json`) or **`SAM3D_GLOBAL_LATENT_PATH`** / **`SAM3D_OBJECT_STEM`** is set.

## Limitations

- SAM3D’s public API here is **single-image**. Extra rendered views are for inspection or a future multi-view fusion path; they are not all fed into one SAM3D forward pass in this module.
