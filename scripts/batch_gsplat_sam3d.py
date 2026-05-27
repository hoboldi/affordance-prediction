#!/usr/bin/env python3
"""
Batch **gsplat → SAM3D** reconstruction with **skip-if-done** semantics (same layout as notebook 10).

Use inside the ``sam3d-pipeline`` Docker container (``/workspace``). Typical patterns::

    # One or more explicit PLYs
    PYTHONPATH=src python scripts/batch_gsplat_sam3d.py \\
        --output_root exports/gsplat_sam3d_runs \\
        data/Seen/train/bag/Gaussian/GS_0017.ply \\
        data/Seen/train/microwave/Gaussian/GS_0056.ply

    # Paths from a text file (one ``.ply`` per line, ``#`` comments allowed)
    PYTHONPATH=src python scripts/batch_gsplat_sam3d.py \\
        --output_root exports/gsplat_sam3d_runs \\
        --splat_paths_file my_splats.txt

    # Random AffordSplat train rows (requires ``Seen/train`` on disk / mount)
    PYTHONPATH=src python scripts/batch_gsplat_sam3d.py \\
        --output_root exports/gsplat_sam3d_runs \\
        --affordsplat_random_n 5 --random_seed 0

Re-runs are skipped when ``<run_dir>/reconstruction/mesh.glb`` and ``meta_prerender.json`` exist and
match the splat. Pass ``--force`` to always invoke SAM3D again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from loguru import logger

from datasets.affordsplat_local_dataset import resolve_affordsplat_root, sample_random_affordsplat_row
from reconstruction.gsplat_sam3d_batch import ensure_sam3d_reconstruction_for_splat
from utils.config import load_config, resolve_path


def _read_splat_paths_file(path: Path) -> list[Path]:
    out: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(resolve_path(Path(line)))
    return out


def _random_affordsplat_splats(
    n: int,
    *,
    seed: int | None,
    subset: str,
    split: str,
    cfg_path: Path | None,
) -> list[Path]:
    cfg = load_config(cfg_path)
    root = resolve_affordsplat_root(cfg)
    if root is None or not (root / subset / split).is_dir():
        raise FileNotFoundError(
            f"No AffordSplat root with {subset}/{split} (set AFFORDANCE_AFFORDSPLAT_ROOT / paths.affordsplat_root)."
        )
    rows: list[Path] = []
    seen: set[str] = set()
    tries = 0
    max_tries = max(256, n * 64)
    while len(rows) < n and tries < max_tries:
        tries += 1
        seed_try = None if seed is None else int(seed) + tries
        row = sample_random_affordsplat_row(
            affordsplat_root=root, seed=seed_try, subset=subset, split=split, cfg=cfg
        )
        if row is None:
            break
        sp = row.splat_path.resolve()
        key = str(sp)
        if key in seen:
            continue
        if not sp.is_file():
            continue
        seen.add(key)
        rows.append(sp)
    if len(rows) < n:
        raise RuntimeError(f"Only collected {len(rows)} unique splats after {tries} tries (wanted {n}).")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Batch gsplat prerender → SAM3D (idempotent)")
    p.add_argument(
        "splat_paths",
        nargs="*",
        type=Path,
        help="One or more 3DGS .ply paths (optional if --splat_paths_file or --affordsplat_random_n)",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("exports/gsplat_sam3d_runs"),
        help="Parent directory for per-splat run folders (default: exports/gsplat_sam3d_runs)",
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config override")
    p.add_argument(
        "--splat_paths_file",
        type=Path,
        default=None,
        help="Text file: one splat .ply per line (# comments ok)",
    )
    p.add_argument(
        "--affordsplat_random_n",
        type=int,
        default=0,
        help="If >0, draw this many random Gaussian rows from AffordSplat (mutually exclusive with explicit paths)",
    )
    p.add_argument("--random_seed", type=int, default=None, help="Base seed for --affordsplat_random_n draws")
    p.add_argument("--subset", type=str, default="Seen", help="AffordSplat subset for random draws")
    p.add_argument("--split", type=str, default="train", help="AffordSplat split for random draws")
    p.add_argument("--force", action="store_true", help="Always run SAM3D (do not skip completed runs)")
    p.add_argument("--max_points", type=int, default=500_000, help="Max Gaussians for gsplat subsample")
    p.add_argument("--reference_view", type=int, default=0, help="Which view_NNN.png to feed SAM3D")
    p.add_argument("--gsplat_seed", type=int, default=None, help="RNG seed for gsplat subsample")
    p.add_argument("--sam3d_seed", type=int, default=42, help="SAM3D torch seed")
    p.add_argument("--no_latent_cache", action="store_true", help="Disable global_latent.pt cache write")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = resolve_path(args.output_root)

    splats: list[Path] = []
    if args.affordsplat_random_n > 0:
        if args.splat_paths or args.splat_paths_file is not None:
            p.error("Use either explicit splat paths / --splat_paths_file OR --affordsplat_random_n, not both.")
        splats = _random_affordsplat_splats(
            args.affordsplat_random_n,
            seed=args.random_seed,
            subset=args.subset,
            split=args.split,
            cfg_path=args.config,
        )
    elif args.splat_paths_file is not None:
        if args.splat_paths:
            p.error("Pass splats either as positional args or via --splat_paths_file, not both.")
        splats = _read_splat_paths_file(resolve_path(args.splat_paths_file))
    else:
        splats = [resolve_path(x) for x in args.splat_paths]

    if not splats:
        p.error("No splat paths: pass .ply files, --splat_paths_file, or --affordsplat_random_n.")

    recon_kw: dict = {
        "max_points": args.max_points,
        "reference_view_index": args.reference_view,
        "gsplat_seed": args.gsplat_seed,
        "sam3d_seed": args.sam3d_seed,
        "cache_global_latent": not args.no_latent_cache,
    }

    n_ran = 0
    n_skip = 0
    for sp in splats:
        if not sp.is_file():
            raise FileNotFoundError(sp)
        r = ensure_sam3d_reconstruction_for_splat(
            sp, output_root=out_root, cfg=cfg, force=args.force, **recon_kw
        )
        if r["status"] == "ran":
            n_ran += 1
            logger.info("SAM3D ran: {} → {}", sp, r["run_dir"])
        else:
            n_skip += 1
            logger.info("Skipped (already done): {} → {}", sp, r["run_dir"])

    logger.info("Done. ran={} skipped={} output_root={}", n_ran, n_skip, out_root)


if __name__ == "__main__":
    main()
