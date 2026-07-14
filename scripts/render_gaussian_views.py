#!/usr/bin/env python3
"""
Render multi-view RGB + depth from a 3D Gaussian Splatting ``.ply``.

**Backends**

- ``gsplat`` (``--backend gsplat``): **true** ellipsoidal Gaussian rasterisation (CUDA + ``gsplat``).
  Use this for object images from a standard Inria / Nerfstudio PLY. Install: ``pip install -e ".[gsplat]"``
  or use the SAM3D Docker image (``[inference]`` already includes ``gsplat``).

- ``preview`` (default): pyrender **centre** preview (points / icospheres) — not full splatting.

Uses the same camera layout as ``configs/default.yaml`` → ``rendering`` (``num_views``, ``fov_deg``, …).
Outputs are suitable as **SAM3D-style RGB inputs** (save masks separately if needed).

Example (full splat, GPU)::

    PYTHONPATH=src python scripts/render_gaussian_views.py \\
        --backend gsplat \\
        --splat_path path/to/your_splat.ply \\
        --output_dir outputs/gaussian_renders/demo

Example (centre preview only, CPU/EGL)::

    PYTHONPATH=src python scripts/render_gaussian_views.py \\
        --backend preview \\
        --splat_path path/to/your_splat.ply \\
        --output_dir outputs/gaussian_renders/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from loguru import logger
from rendering.gaussian_gsplat_renderer import render_gaussian_splat_gsplat_views
from rendering.gaussian_point_renderer import render_gaussian_splat_views
from rendering.renderer import build_render_config
from utils.config import load_config, resolve_path


def _serialize_camera(intrinsics: np.ndarray, c2w: np.ndarray) -> dict:
    return {
        "intrinsics": intrinsics.astype(float).tolist(),
        "camera_to_world": c2w.astype(float).tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Render 2D views from a 3DGS PLY (gsplat or centre preview)")
    p.add_argument("--splat_path", type=Path, required=True, help="Path to 3DGS-style .ply")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to write PNG / NPY / JSON")
    p.add_argument(
        "--backend",
        choices=("preview", "gsplat"),
        default="preview",
        help="gsplat: true Gaussian rasterisation (CUDA). preview: pyrender centre points/icospheres.",
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--max_points", type=int, default=200_000, help="Max Gaussians to draw (subsampled)")
    p.add_argument(
        "--no_normalize_scene",
        action="store_true",
        help="Keep original world coordinates (cameras still use rendering.camera_radius)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling Gaussians")
    p.add_argument(
        "--legacy_gl_gsplat_camera",
        action="store_true",
        help="gsplat only: skip OpenGL→OpenCV camera conversion (old behaviour; real 3DGS PLYs often go black).",
    )
    p.add_argument(
        "--preview_mode",
        choices=("auto", "spheres", "points"),
        default="auto",
        help="auto: icosphere blobs if N<=sphere_merge_max (visible in pyrender); else GL_POINTS",
    )
    p.add_argument(
        "--sphere_merge_max",
        type=int,
        default=2048,
        help="With --preview_mode auto, use icospheres when splat has at most this many centres",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    out = resolve_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splat = resolve_path(args.splat_path)
    if not splat.is_file():
        raise FileNotFoundError(f"Missing splat PLY: {splat}")

    mrc = build_render_config(cfg)
    logger.info(
        "Rendering {} views from {} (backend={}, max_points={}, normalize={})",
        mrc.num_views,
        splat,
        args.backend,
        args.max_points,
        not args.no_normalize_scene,
    )

    if args.backend == "gsplat":
        views = render_gaussian_splat_gsplat_views(
            splat,
            mrc,
            max_points=args.max_points,
            normalize_scene=not args.no_normalize_scene,
            seed=args.seed,
            convert_pyrender_camera_to_gsplat=not args.legacy_gl_gsplat_camera,
        )
    else:
        views = render_gaussian_splat_views(
            splat,
            mrc,
            max_points=args.max_points,
            normalize_scene=not args.no_normalize_scene,
            seed=args.seed,
            preview_mode=args.preview_mode,
            sphere_merge_max=args.sphere_merge_max,
        )

    meta_views: list[dict] = []
    for i, view in enumerate(views):
        stem = f"view_{i:03d}"
        Image.fromarray(view.rgb).save(out / f"{stem}.png")
        np.save(out / f"{stem}_depth.npy", view.depth)
        meta_views.append(
            {
                "index": i,
                "rgb": f"{stem}.png",
                "depth": f"{stem}_depth.npy",
                **_serialize_camera(view.intrinsics, view.camera_pose),
            }
        )

    manifest = {
        "splat_path": str(splat),
        "num_views": len(views),
        "image_size": mrc.image_size,
        "fov_deg": mrc.fov_deg,
        "camera_radius": mrc.camera_radius,
        "elevation_deg": mrc.elevation_deg,
        "elevation_min_deg": mrc.elevation_min_deg,
        "elevation_max_deg": mrc.elevation_max_deg,
        "max_points": args.max_points,
        "normalize_scene": not args.no_normalize_scene,
        "backend": args.backend,
        "views": meta_views,
    }
    if args.backend == "preview":
        manifest["preview_mode"] = args.preview_mode
        manifest["sphere_merge_max"] = args.sphere_merge_max
    (out / "cameras.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote {} views under {}", len(views), out)


if __name__ == "__main__":
    main()
