# Exported artifacts (visible in the IDE)

Unlike `outputs/`, this tree is **not** listed in `.gitignore`, so your editor’s file explorer should show it after the notebook runs.

| Path | Writer |
|------|--------|
| `gaussian_splat/` | `notebooks/03_rendering_gaussian_splat.ipynb` — PNG/JPEG frames from gsplat (or preview fallback) |

You can delete generated `*.png` / `*.jpg` here anytime; they are not required for CI.
