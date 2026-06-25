# Example 3D Gaussian Splatting PLY

`tiny_gaussians.ply` is a **synthetic** ball of **400** Gaussians in the usual **Inria / Nerfstudio** property layout (`x,y,z`, `f_dc_*`, `opacity`, `scale_*`, `rot_*`). Positions use a **Fibonacci sphere** (golden-angle spiral in `y`); an earlier build mistakenly used `golden * i / N` for azimuth, which collapsed points onto a **helix / line**—regenerate this file if you copied an old revision.

Used by `tests/test_gaussian_rendering.py` and `tests/test_gaussian_gsplat_rendering.py`. For **true** RGB from this file (ellipsoidal splatting), use `render_gaussian_splat_gsplat_views` / `scripts/render_gaussian_views.py --backend gsplat` with CUDA + `gsplat` (see `pyproject.toml` optional `[gsplat]`). The notebook `notebooks/03_rendering_gaussian_splat.ipynb` prefers that path when the GPU stack is available.

Replace with a real **3DGS export** from your data pipeline (e.g. AffordSplat / COLMAP) for experiments.
