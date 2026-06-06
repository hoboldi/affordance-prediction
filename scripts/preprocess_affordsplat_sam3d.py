#!/usr/bin/env python3
"""
Pre-processing pipeline: AffordSplat → SAM3D → training manifest.

Only processes splats that have at least one affordance annotation (``GS_anno_*.ply``).
For every such unique Gaussian this script:
  1. Ensures SAM3D artifacts exist under ``output_root`` (runs SAM3D if missing).
  2. Writes a ``manifest.jsonl`` where each line is one (splat, verb) training row
     pointing at the ``sam3d_reconstruction_dir`` that contains
     ``slat_vertex_features.pt``, ``global_latent.pt``, ``mesh.glb``, etc.

The manifest can then be consumed by ``DataRootDataset`` for training.

Run inside the ``sam3d-pipeline`` Docker container (CUDA + gsplat + SAM3D)::

    PYTHONPATH=src python scripts/preprocess_affordsplat_sam3d.py \\
        --output_root exports/gsplat_sam3d_runs \\
        --manifest_out data/manifests/train_sam3d.jsonl \\
        --subset Seen --split train

Use ``--conditioning_verb grasp`` (or any short token) for **one manifest row per annotated splat**
with that fixed ``verb`` string; vertex labels still come from one ``GS_anno_*.ply`` (matching
folder name if present, else the first sorted affordance folder).

Resume safely: already-finished runs are detected by ``sam3d_run_layout_ready``
and skipped. Use ``--force`` to always re-run SAM3D.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tqdm import tqdm

from datasets.affordsplat_local_dataset import (
    AffordSplatLocalRow,
    iter_affordsplat_local_rows,
    resolve_affordsplat_root,
)
from reconstruction.gsplat_sam3d_batch import (
    ensure_sam3d_reconstruction_for_splat,
)
import torch

from reconstruction.sam3d_wrapper import SAM3DWrapper, sam3d_environment
from utils.config import load_config, project_root, resolve_path


@contextlib.contextmanager
def _silence():
    """Redirect stdout + stderr to /dev/null at the fd level (suppresses C/CUDA output too)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(devnull_fd)


def _iter_unique_annotated_splats(
    rows: list[AffordSplatLocalRow],
) -> list[tuple[Path, list[AffordSplatLocalRow]]]:
    """
    Group rows by splat path, keeping only splats that have at least one affordance annotation.
    Preserves first-seen order.
    """
    seen: dict[str, list[AffordSplatLocalRow]] = {}
    for row in rows:
        if row.affordance_gs_anno_path is None:
            continue
        key = str(row.splat_path.resolve())
        seen.setdefault(key, []).append(row)
    return [(Path(k), v) for k, v in seen.items()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="AffordSplat → SAM3D pre-processing: generate artifacts and write manifest"
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("exports/gsplat_sam3d_runs"),
        help="Parent directory for per-splat SAM3D run folders (use /workspace/data/sam3d inside Docker)",
    )
    p.add_argument(
        "--manifest_out",
        type=Path,
        required=True,
        help="Path for the output manifest.jsonl (will be created / overwritten)",
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config override")
    p.add_argument("--subset", type=str, default="Seen", help="AffordSplat subset (Seen / UnSeen)")
    p.add_argument("--split", type=str, default=None, help="train / val / test (default: all)")
    p.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Limit to these categories (default: all)",
    )
    p.add_argument(
        "--conditioning_verb",
        type=str,
        default=None,
        help="If set (e.g. grasp), one row per annotated splat with this verb; see iter_affordsplat_local_rows",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N splats (useful for dry runs)")
    p.add_argument("--force", action="store_true", help="Re-run SAM3D even if artifacts exist")
    p.add_argument("--manifest_only", action="store_true", help="Skip SAM3D — just write manifest from already-completed runs")
    p.add_argument("--verbose", action="store_true", help="Show all SAM3D output (useful for nohup logs)")
    p.add_argument("--max_points", type=int, default=500_000)
    p.add_argument("--reference_view", type=int, default=0)
    p.add_argument("--gsplat_seed", type=int, default=None)
    p.add_argument("--sam3d_seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config(args.config)
    root = resolve_affordsplat_root(cfg)
    if root is None:
        p.error(
            "AffordSplat root not found. Set AFFORDANCE_AFFORDSPLAT_ROOT or paths.affordsplat_root."
        )

    cat_set = set(args.categories) if args.categories else None
    rows = list(
        iter_affordsplat_local_rows(
            root,
            subset=args.subset,
            split=args.split,
            categories=cat_set,
            conditioning_verb=args.conditioning_verb,
        )
    )
    if not rows:
        p.error(f"No rows found under {root}/{args.subset} (split={args.split!r})")

    out_root = resolve_path(args.output_root)
    recon_kw = {
        "max_points": args.max_points,
        "reference_view_index": args.reference_view,
        "gsplat_seed": args.gsplat_seed,
        "sam3d_seed": args.sam3d_seed,
    }

    splat_groups = _iter_unique_annotated_splats(rows)
    if args.limit is not None:
        splat_groups = list(splat_groups)[: args.limit]
    n_total_rows = sum(len(g) for _, g in splat_groups)
    tqdm.write(
        f"Annotated splats: {len(splat_groups)}  ({n_total_rows} (splat, verb) rows)"
        f"  from {args.subset}/{args.split or 'all'}"
    )

    splat_to_recon_dir: dict[str, Path] = {}

    if args.manifest_only:
        from reconstruction.gsplat_sam3d_batch import sam3d_run_layout_ready, splat_run_dir_slug
        n_found = n_missing = 0
        for splat_path, _group_rows in tqdm(splat_groups, unit="splat", dynamic_ncols=True):
            run_dir = out_root / splat_run_dir_slug(splat_path)
            if sam3d_run_layout_ready(run_dir):
                splat_to_recon_dir[str(splat_path.resolve())] = run_dir / "reconstruction"
                n_found += 1
            else:
                n_missing += 1
        tqdm.write(f"Found {n_found} completed runs, {n_missing} missing.")
    else:
        recon_cfg = cfg.get("reconstruction") or {}
        sam3d_config_path = recon_cfg.get(
            "sam3d_config", "sam-3d-objects/checkpoints/hf/pipeline.yaml"
        )
        compile_model = bool(recon_cfg.get("compile", False))
        resolved_config = resolve_path(str(sam3d_config_path), root=project_root())

        n_ran = n_skip = n_fail = 0
        silence = _silence if not args.verbose else contextlib.nullcontext

        with sam3d_environment(project_root()):
            tqdm.write("Loading SAM3D model ...")
            with silence():
                wrapper = SAM3DWrapper(resolved_config, compile_model=compile_model)
            tqdm.write("SAM3D model loaded.")

            bar = tqdm(splat_groups, unit="splat", dynamic_ncols=True)
            for splat_path, _group_rows in bar:
                bar.set_description(splat_path.stem[:40])
                caught: BaseException | None = None
                with silence():
                    try:
                        result = ensure_sam3d_reconstruction_for_splat(
                            splat_path,
                            output_root=out_root,
                            cfg=cfg,
                            wrapper=wrapper,
                            force=args.force,
                            **recon_kw,
                        )
                    except Exception as exc:
                        caught = exc

                gc.collect()
                torch.cuda.empty_cache()

                if caught is not None:
                    tqdm.write(f"FAILED {splat_path}:")
                    tqdm.write(traceback.format_exc())
                    n_fail += 1
                    continue

                recon_dir = Path(result["reconstruction_dir"])
                splat_to_recon_dir[str(splat_path.resolve())] = recon_dir
                if result["status"] == "ran":
                    n_ran += 1
                else:
                    n_skip += 1

        tqdm.write(
            f"Done: {n_ran} ran, {n_skip} skipped, {n_fail} failed out of {len(splat_groups)} splats."
        )

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_recon = 0

    # Only write rows that have annotations and a finished reconstruction
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if row.affordance_gs_anno_path is None:
                continue
            recon_dir = splat_to_recon_dir.get(str(row.splat_path.resolve()))
            if recon_dir is None:
                skipped_no_recon += 1
                continue
            entry: dict = {
                "sample_id": row.sample_id,
                "verb": row.verb,
                "split": row.split,
                "splat_path": str(row.splat_path),
                "sam3d_reconstruction_dir": str(recon_dir),
                "vertex_affordance_path": str(row.affordance_gs_anno_path),
            }
            f.write(json.dumps(entry) + "\n")
            written += 1

    tqdm.write(
        f"Manifest written: {written} rows → {manifest_path}"
        + (f"  ({skipped_no_recon} skipped — no reconstruction)" if skipped_no_recon else "")
    )


if __name__ == "__main__":
    main()
