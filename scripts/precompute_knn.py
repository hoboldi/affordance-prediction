"""Precompute per-vertex kNN graphs for the GNN affordance backbone.

For each UNIQUE object (reconstruction dir) referenced by a manifest, load the per-vertex positions,
build a scipy cKDTree, query the k+1 nearest neighbours, drop the self-edge, and save the resulting
(V, k) int32 index tensor to ``<recon_dir>/vertex_knn_k{K}.pt``.

Idempotent: an object whose output file already exists is skipped. The GNN can also build the graph
on-the-fly from positions, so this precompute is purely a training-time speedup (build once, reuse
every epoch).

Usage:
    PYTHONPATH=src python scripts/precompute_knn.py --manifest /path/to/manifest.jsonl --k 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if (_root / "src").is_dir():
    sys.path.insert(0, str(_root / "src"))

import numpy as np
import torch
from scipy.spatial import cKDTree

from datasets.data_root_dataset import (
    _load_vertex_positions,
    load_manifest_rows,
    resolve_data_root,
)
from utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute per-vertex kNN graphs for the GNN backbone")
    p.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl")
    p.add_argument("--k", type=int, default=8, help="Number of neighbours (self-edge excluded)")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    data_root = resolve_data_root(cfg)
    rows = load_manifest_rows(args.manifest, data_root=data_root)

    # Unique reconstruction dirs (objects), preserving first-seen order.
    recon_dirs: list[Path] = []
    seen: set[str] = set()
    for r in rows:
        d = r.sam3d_reconstruction_dir
        if d is None:
            continue
        key = str(d)
        if key not in seen:
            seen.add(key)
            recon_dirs.append(d)

    out_name = f"vertex_knn_k{args.k}.pt"
    print(f"{len(recon_dirs)} unique objects in {args.manifest} (data_root={data_root})")
    print(f"Writing {out_name} (k={args.k}) per object …")

    n_done = n_skip = n_miss = 0
    for i, recon_dir in enumerate(recon_dirs):
        out_path = recon_dir / out_name
        if out_path.is_file():
            n_skip += 1
        else:
            pos = _load_vertex_positions(recon_dir)
            if pos is None:
                n_miss += 1
                print(f"  WARN no positions/mesh for {recon_dir} — skipped")
            else:
                xyz = pos.detach().cpu().numpy()
                tree = cKDTree(xyz)
                _, idx = tree.query(xyz, k=args.k + 1)  # first hit is the point itself
                knn = np.asarray(idx[:, 1:], dtype=np.int32)  # (V, k), drop self
                torch.save(torch.from_numpy(knn), out_path)
                n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(recon_dirs)}] computed={n_done} skipped={n_skip} missing={n_miss}")

    print(f"Done: computed={n_done} skipped(existing)={n_skip} missing={n_miss}")


if __name__ == "__main__":
    main()
