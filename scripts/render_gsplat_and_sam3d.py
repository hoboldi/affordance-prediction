#!/usr/bin/env python3
"""
True **gsplat** multi-view render from a 3DGS ``.ply``, then **SAM3D** on ``view_000`` (or ``--reference_view``).

**You cannot use this outside the container** in a supported way: SAM3D + ``gsplat`` expect the
pinned CUDA / torch / ``flash-attn`` / ``sam-3d-objects`` layout from the ``sam3d-pipeline``
Docker image. Run this script inside ``docker compose run --rm sam3d-pipeline bash`` (repo at
``/workspace``); see ``docker/README.md``.

Writes::

    <run_dir>/sam3d_dataset/images, masks, meta_prerender.json
    <run_dir>/reconstruction/{mesh.glb, gaussian.ply, shape_latent.pt, …}

Optional: cache ``global_latent.pt`` under ``paths.cache_root/sam3d/<stem>/`` (see ``configs/default.yaml``).

Example (inside the container, from ``/workspace``)::

    PYTHONPATH=src python scripts/render_gsplat_and_sam3d.py \\
        --splat_path data/Seen/train/bag/Gaussian/GS_0017.ply \\
        --run_dir exports/gsplat_sam3d/bag_gs0017
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from loguru import logger

from reconstruction.gsplat_to_sam3d import gsplat_ply_to_sam3d_reconstruction
from utils.config import load_config, resolve_path


def main() -> None:
    p = argparse.ArgumentParser(description="gsplat views → SAM3D reconstruction")
    p.add_argument("--splat_path", type=Path, required=True, help="Input 3DGS .ply")
    p.add_argument("--run_dir", type=Path, required=True, help="Output run directory")
    p.add_argument("--config", type=Path, default=None, help="YAML config override")
    p.add_argument("--object_stem", type=str, default=None, help="SAM3D / cache stem (default from PLY + view id)")
    p.add_argument("--reference_view", type=int, default=0, help="Which view_NNN.png to feed SAM3D")
    p.add_argument("--max_points", type=int, default=500_000, help="Max Gaussians for gsplat subsample")
    p.add_argument(
        "--gsplat_seed",
        type=int,
        default=None,
        help="RNG seed for gsplat subsample (omit for random)",
    )
    p.add_argument("--sam3d_seed", type=int, default=42, help="SAM3D torch seed")
    p.add_argument("--no_latent_cache", action="store_true", help="Disable global_latent.pt cache write")
    args = p.parse_args()

    cfg = load_config(args.config)
    splat = resolve_path(args.splat_path)
    if not splat.is_file():
        raise FileNotFoundError(f"Missing splat: {splat}")

    run_dir = resolve_path(args.run_dir)
    logger.info("Running gsplat → SAM3D: splat={} → {}", splat, run_dir)
    out = gsplat_ply_to_sam3d_reconstruction(
        splat,
        run_dir,
        cfg=cfg,
        object_stem=args.object_stem,
        max_points=args.max_points,
        gsplat_seed=args.gsplat_seed,
        reference_view_index=args.reference_view,
        sam3d_seed=args.sam3d_seed,
        cache_global_latent=not args.no_latent_cache,
    )
    logger.info("Reconstruction dir: {}", out["reconstruction_dir"])
    if "global_latent_path" in out:
        logger.info("Cached global latent: {}", out["global_latent_path"])
    run_root = Path(out["reconstruction_dir"]).parent
    logger.info(
        "Wire rendering/VLM/projection: export SAM3D_RUN_DIR={} (see notebooks 02,04,05,06).",
        run_root,
    )


if __name__ == "__main__":
    main()
