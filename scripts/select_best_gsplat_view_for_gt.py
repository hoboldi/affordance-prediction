#!/usr/bin/env python3
"""
Select a single best 2D RGB view from a 3DGS ``.ply`` by scoring unprojected depth vs a GT point cloud.

Requires **CUDA** + **gsplat** (same stack as ``render_gaussian_splat_gsplat_views``). Intended for
the ``sam3d-pipeline`` Docker image when SAM3D deps are available.

Example with AffordSplat affordance upweighting::

    PYTHONPATH=src python scripts/select_best_gsplat_view_for_gt.py \\
        --splat_path data/Seen/train/bag/Gaussian/GS_0017.ply \\
        --gt_path data/Seen/train/bag/PointCloud/PC_0017.ply \\
        --affordance_gt_path data/Seen/train/bag/contain/GS_anno_0017.ply \\
        --affordance_score_weight 0.35 \\
        --output_png exports/best_view/best_rgb.png \\
        --meta_json exports/best_view/meta.json

Minimal smoke (no affordance term)::

    PYTHONPATH=src python scripts/select_best_gsplat_view_for_gt.py \\
        --splat_path examples/gaussian_splat/tiny_gaussians.ply \\
        --gt_path path/to/gt_vertices.npy \\
        --output_png exports/best_view/best_rgb.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from loguru import logger

from rendering.gsplat_viewpoint_selection import (
    ViewSelectionConfig,
    export_best_view_png,
    select_best_gsplat_view_for_gt,
)
from utils.config import resolve_path


def main() -> None:
    p = argparse.ArgumentParser(description="Best 3DGS view vs GT point cloud (depth alignment)")
    p.add_argument("--splat_path", type=Path, required=True)
    p.add_argument("--gt_path", type=Path, required=True)
    p.add_argument("--output_png", type=Path, required=True)
    p.add_argument("--meta_json", type=Path, default=None, help="Defaults to <output_png>.json")
    p.add_argument("--num_views", type=int, default=3)
    p.add_argument(
        "--orbit_azimuth_offsets_deg",
        type=str,
        default="-45,0,45",
        help="Comma-separated azimuth offsets (degrees) on the ring — length must match --num_views. "
        "Use 'uniform' for an evenly spaced full-ring sweep instead.",
    )
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--camera_radius", type=float, default=2.0)
    p.add_argument("--elevation_deg", type=float, default=42.5)
    p.add_argument("--elevation_min_deg", type=float, default=25.0)
    p.add_argument("--elevation_max_deg", type=float, default=55.0)
    p.add_argument("--max_gsplat_points", type=int, default=500_000)
    p.add_argument("--unproject_stride", type=int, default=2)
    p.add_argument("--score_max_render", type=int, default=4096)
    p.add_argument("--score_max_gt", type=int, default=8192)
    p.add_argument(
        "--affordance_gt_path",
        type=Path,
        default=None,
        help="Optional 3D affordance points (e.g. AffordSplat …/contain/GS_anno_<id>.ply); "
        "same frame as full GT. Used when --affordance_score_weight > 0.",
    )
    p.add_argument(
        "--affordance_score_weight",
        type=float,
        default=0.0,
        help="Multiply affordance idw-kNN score and add to base GT score (0 = disabled).",
    )
    p.add_argument("--score_max_affordance", type=int, default=4096)
    p.add_argument(
        "--strict_geometry_rejects",
        action="store_true",
        help="Enable elongation + NN outlier reject gates (default off; often rejects all views on sparse GT).",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    def _parse_azimuth_offsets(raw: str) -> tuple[float, ...] | None:
        t = raw.strip().lower()
        if t in ("uniform", "none", ""):
            return None
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        return tuple(float(x) for x in parts)

    az_off = _parse_azimuth_offsets(args.orbit_azimuth_offsets_deg)
    if az_off is not None and len(az_off) != args.num_views:
        p.error(
            f"--orbit_azimuth_offsets_deg has {len(az_off)} values but --num_views is {args.num_views}; "
            "they must match, or use --orbit_azimuth_offsets_deg uniform"
        )

    splat = resolve_path(args.splat_path)
    gt = resolve_path(args.gt_path)
    out_png = resolve_path(args.output_png)
    meta_path = (
        resolve_path(args.meta_json)
        if args.meta_json is not None
        else out_png.with_suffix(".json")
    )
    aff_path = resolve_path(args.affordance_gt_path) if args.affordance_gt_path is not None else None
    cfg = ViewSelectionConfig(
        num_views=args.num_views,
        image_size=args.image_size,
        fov_deg=args.fov_deg,
        camera_radius=args.camera_radius,
        elevation_deg=args.elevation_deg,
        elevation_min_deg=args.elevation_min_deg,
        elevation_max_deg=args.elevation_max_deg,
        max_gsplat_points=args.max_gsplat_points,
        unproject_stride=args.unproject_stride,
        score_max_render=args.score_max_render,
        score_max_gt=args.score_max_gt,
        seed=args.seed,
        affordance_gt_path=aff_path,
        affordance_score_weight=args.affordance_score_weight,
        score_max_affordance=args.score_max_affordance,
        use_strict_geometry_rejects=args.strict_geometry_rejects,
        orbit_azimuth_offsets_deg=az_off,
    )
    logger.info("Selecting best view: {} views, seed={}", cfg.num_views, cfg.seed)
    result = select_best_gsplat_view_for_gt(splat, gt, cfg)
    logger.info(
        "Best view index={} combined={:.6g} base={:.6g} aff={} chamfer_mean_sq={:.6g}",
        result.best_index,
        result.scores[result.best_index],
        result.base_scores[result.best_index],
        result.affordance_scores[result.best_index],
        result.chamfers[result.best_index],
    )
    export_best_view_png(result, out_png, meta_json=meta_path)
    logger.info("Wrote {} and {}", out_png.resolve(), meta_path.resolve())


if __name__ == "__main__":
    main()
