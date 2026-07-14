"""Prepare CO3D v2 for the GEAL-pseudolabel pipeline (REAL images).

Scans CO3D's per-category sequences, keeps only categories that overlap GEAL's taxonomy
(``configs/co3d.yaml`` ``category_map``), stages an ``images/`` + ``masks/`` directory for
``scripts/generate_sam3d.py`` (one canonical frame + its CO3D mask per sequence), and writes a
training manifest with ``object_class`` + ``verb`` + per-sample ``split`` (``val_categories`` -> "val").

Replaces the earlier synthetic-render ingestion.

Next steps (run later, need disk/GPU):
    python scripts/generate_sam3d.py --dataset_dir <data_root>/<staged_dir> --output_dir <data_root>/<recon_output_dir>
    python scripts/generate_geal_pseudolabels.py --manifest <data_root>/<manifest> --ckpt external/geal/ckpt/piad_seen.pt
    python scripts/train_affordance.py --manifest <data_root>/<manifest>.pseudolabeled.jsonl

Usage:
    python scripts/prepare_co3d.py --config configs/co3d.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import yaml

from datasets.co3d import discover_samples, stage_sam3d_inputs, write_manifest

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare CO3D inputs + manifest")
    p.add_argument("--config", default="configs/co3d.yaml", help="CO3D config yaml")
    p.add_argument("--data_root", default=None, help="Override paths.data_root from the config")
    p.add_argument("--views_per_object", type=int, default=None, help="Override views per object")
    p.add_argument("--max_objects_per_category", type=int, default=None, help="Cap sequences kept per category")
    p.add_argument("--no_stage", action="store_true", help="Only write the manifest; skip image/mask staging")
    p.add_argument("--limit", type=int, default=None, help="Keep at most N samples (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    co = cfg.get("co3d", {})

    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else (_REPO_ROOT / cfg["paths"]["data_root"]).resolve()
    )
    co3d_root = data_root / co.get("co3d_root", "source")
    dataset_dir = data_root / co.get("staged_dir", "sam3d_inputs")
    recon_output_dir = data_root / co.get("recon_output_dir", "reconstructions")
    manifest_path = data_root / co.get("manifest_filename", "manifest.jsonl")
    views = args.views_per_object if args.views_per_object is not None else int(co.get("views_per_object", 1))
    max_obj = (
        args.max_objects_per_category
        if args.max_objects_per_category is not None
        else co.get("max_objects_per_category")
    )

    samples = discover_samples(
        co3d_root,
        co.get("category_map", {}),
        co.get("affordance_map", {}),
        views_per_object=views,
        frame_selection=str(co.get("frame_selection", "middle")),
        max_objects_per_category=max_obj,
        val_categories=co.get("val_categories"),
    )
    if args.limit is not None:
        samples = samples[: args.limit]

    by_class: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for s in samples:
        by_class[s.geal_class] = by_class.get(s.geal_class, 0) + 1
        by_split[s.split] = by_split.get(s.split, 0) + 1
    log.info("Discovered %d samples across %d GEAL classes: %s", len(samples), len(by_class), by_class)
    log.info("Split counts: %s", by_split)
    if not samples:
        log.warning("No samples found under %s — check CO3D root and category_map folder names.", co3d_root)
        return

    if not args.no_stage:
        log.info("Staging images/masks under %s ...", dataset_dir)
        stage_sam3d_inputs(samples, dataset_dir)

    write_manifest(
        samples,
        manifest_path,
        data_root=data_root,
        dataset_dir=dataset_dir,
        recon_output_dir=recon_output_dir,
    )
    log.info("Wrote manifest with %d rows -> %s", len(samples), manifest_path)


if __name__ == "__main__":
    main()
