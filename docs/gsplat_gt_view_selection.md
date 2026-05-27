# 3DGS viewpoint selection vs GT point cloud

Deterministic pipeline (see `rendering/gsplat_viewpoint_selection.py`, CLI `scripts/select_best_gsplat_view_for_gt.py`, notebook `notebooks/11_gsplat_view_selection_vs_gt.ipynb` for a **grid of all views + winner**). The notebook is **Run All**–oriented: it resolves the AffordSplat root from config/env, draws a random **`Seen/train`** row with a matching **`PointCloud/PC_<id>.ply`** (retries), or falls back to the bundled tiny splat; optional knobs are `FORCE_TINY_TOY` and `RANDOM_SAMPLE_SEED`. Sampling uses `datasets.sample_random_affordsplat_row` (uniform over the same stream as the local dataset).

1. **Orbit** — by default **three** cameras on the ring at azimuths **`orbit_azimuth_offsets_deg`** (**−45°, 0°, +45°** vs the same zero direction as a uniform sweep’s first camera). Set **`orbit_azimuth_offsets_deg=None`** and choose **`num_views`** for a **uniform** full-ring sweep. The pole comes from `MeshRenderConfig` / `ViewSelectionConfig`: **world +Y** by default, optional **`orbit_ring_rotation_deg`** about **`orbit_ring_rotation_axis`** (view selection defaults **+90° about +X** for common splat-frame tilt), then optional **`orbit_axis`** / **`orbit_axis_mode`** (`pca_min` / `pca_max`). Pitch is clamped to **`[elevation_min_deg, elevation_max_deg]`** intersected with the global policy in `rendering/camera_sampling.py` (**25°–60°**).
2. **Render + unproject** — `gsplat` `RGB+ED` depth; pixels unprojected with pinhole **K** (Z-buffer assumption) to camera space, then transformed to **normalized scene coordinates** using the same center/scale as the gsplat mean normalization (subsampled PLY, fixed `seed`).
3. **Score** — inverse-distance **k=3** NN mean (higher is better) between unprojected depth points and subsampled **full-object GT** points, plus **one-way Chamfer** mean squared distance for diagnostics. The winning view is the highest combined score; **near-equal scores** (within a small relative band of the max) break ties by **lower Chamfer**, then lower view index. **Rejections (default):** only **very low valid-depth coverage** (`min_valid_frac`) and **non-finite Chamfer**. Optional **`use_strict_geometry_rejects=True`** also enforces covariance elongation and NN outlier gates (intended for dense GT / sanity checks; often rejects every view for **sparse `PC_*.ply`** or **depth fan** geometry).
4. **Affordance term (optional)** — if `affordance_gt_path` is set and `affordance_score_weight > 0`, add `weight × idw_kNN(render, affordance_points)` using the same normalization as GT. Typical source: AffordSplat `Seen/.../<category>/<verb>/GS_anno_<id>.ply` (3D points for that affordance). This **prefers camera views whose depth agrees with the desired affordance region**, not only the full object PC.
5. **Export** — best RGB PNG and JSON (`camera_to_world`, `intrinsics`, per-view `scores` = combined, `base_scores`, `affordance_scores`, rejection reasons). JSON also includes **`scene_normalize_center`** and **`scene_normalize_scale`** (same isotropic normalize as depth scoring) so downstream notebooks can align meshes with GT. Non-finite floats are written as JSON `null`.

## Requirements

- **CUDA** + **`gsplat`** (same as `render_gaussian_splat_gsplat_views`). Prefer the `sam3d-pipeline` Docker image.
- **GT** — `.npy` shaped `(N, 3)` or a mesh-like file readable by `trimesh` (vertices used as points).

## Coordinate frame

**GT must live in the same world frame as the 3DGS means** before the script’s isotropic normalization (centering + uniform scale from the splat). If GT comes from a mesh in another frame, rigidly align it to the splat offline.

The optional **affordance** point file must use that **same** frame (AffordSplat `GS_anno_*.ply` is authored against the same object Gaussian as `GS_*.ply`).

## Depth convention

Depth is treated as **positive camera-space Z** along rays through the pinhole model. If a dataset’s ED channel uses a different convention, unprojection and scores may be wrong — adjust `unproject_depth_zbuffer` if needed.
