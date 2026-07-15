#!/usr/bin/env python3
"""Download the ReVerb dataset (CO3D single-image SAM3D reconstructions + human affordance labels)
from the Hugging Face Hub.

The dataset ships the reconstructed meshes and the per-vertex human annotations; per-vertex features
(DINOv2, geometry) are regenerated locally with ``scripts/generate_vertex_dino.py`` and
``scripts/precompute_geom.py``. See ``docs/data_layout.md`` for the on-disk layout.

Usage::

    python scripts/download_dataset.py                        # -> <repo>/data
    python scripts/download_dataset.py --local-dir /path/to/data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "MattisKr/co3d-sam3d-affordance"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download the ReVerb dataset from the Hugging Face Hub.",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="destination directory (default: <repo>/data)",
    )
    p.add_argument("--repo-id", default=REPO_ID, help=f"HF dataset repo (default: {REPO_ID})")
    args = p.parse_args()

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download  # imported late so --help works without the dep

    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.repo_id} -> {args.local_dir}", flush=True)
    snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=args.local_dir)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
