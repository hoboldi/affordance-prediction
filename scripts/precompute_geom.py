"""Precompute per-vertex 3D geometry features for the affordance model (a geometry-only channel).

For each UNIQUE object (reconstruction dir) referenced by a manifest, using ``vertex_positions``,
``vertex_normals`` and the precomputed ``vertex_knn_k{K}.pt`` neighbour index (V, k), compute a
[V, 5] float32 ``vertex_geom.pt`` (FULLY VECTORIZED over vertices — gathers neighbours via the (V,k)
index, no per-vertex python loop):

    ch0 height_up  = position y-coord (gravity = +y; the render plots xyz[:,1] vertical),
                     per-object min-max normalized to [0, 1].
    ch1 concavity  = mean_j (pos[j] - pos[i]) . normal[i]   over the kNN j of i.
                     (>0 => neighbours sit on the +normal side: a concave pocket/interior.)
    ch2 curvature  = mean_j (1 - normal[i].normal[j])       over the kNN j of i.
                     (normal disagreement: high on edges/corners, ~0 on flats.)
    ch3 normal_up  = normal[i] . (0, 1, 0).                  (upward-facing-ness; seats, table tops.)
    ch4 radial     = ||pos[i] - centroid|| / max_radius.     (extremity; rims, handles, tips.)

ch1-ch4 are standardized per object (zero-mean / unit-std, eps-guarded) to ~unit scale so they sit
comfortably alongside the (LayerNorm'd) learned feature channels. ch0 is left in [0,1] (an absolute
height cue, intentionally on its own scale).

Idempotent (``--skip_existing``); prints progress every 200 objects.

Usage:
    PYTHONPATH=src python scripts/precompute_geom.py \
        --manifest /home/datasets/customDatasets/cmr2/manifest.finepatch.jsonl --k 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if (_root / "src").is_dir():
    sys.path.insert(0, str(_root / "src"))

import torch

from datasets.data_root_dataset import (
    _load_vertex_positions,
    load_manifest_rows,
    resolve_data_root,
)
from utils.config import load_config

GEOM_DIM = 5


def compute_geom(pos: torch.Tensor, normals: torch.Tensor, knn: torch.Tensor) -> torch.Tensor:
    """[V, 5] geometry features, vectorized over vertices via the (V, k) neighbour gather.

    pos:     (V, 3) float  vertex positions (centered + unit-sphere normalized upstream)
    normals: (V, 3) float  per-vertex unit normals
    knn:     (V, k) long    neighbour indices (self-edge already dropped)
    """
    pos = pos.float()
    normals = torch.nn.functional.normalize(normals.float(), dim=-1, eps=1e-8)
    knn = knn.long()
    V = pos.shape[0]

    pj = pos[knn]            # (V, k, 3) neighbour positions
    nj = normals[knn]        # (V, k, 3) neighbour normals
    ni = normals.unsqueeze(1)  # (V, 1, 3)

    # ch0 height_up: y-coord, per-object min-max -> [0,1]
    y = pos[:, 1]
    y_min, y_max = y.min(), y.max()
    height_up = (y - y_min) / (y_max - y_min + 1e-8)

    # ch1 concavity: mean over kNN of (pos[j]-pos[i]) . normal[i]
    diff = pj - pos.unsqueeze(1)                       # (V, k, 3)
    concavity = (diff * ni).sum(-1).mean(dim=1)        # (V,)

    # ch2 curvature: mean over kNN of (1 - n_i . n_j)
    curvature = (1.0 - (ni * nj).sum(-1)).mean(dim=1)  # (V,)

    # ch3 normal_up: n_i . (0,1,0)
    normal_up = normals[:, 1]                          # (V,)

    # ch4 radial: ||pos - centroid|| / max_radius
    centroid = pos.mean(dim=0, keepdim=True)
    r = (pos - centroid).norm(dim=-1)                  # (V,)
    radial = r / (r.max() + 1e-8)

    geom = torch.stack([height_up, concavity, curvature, normal_up, radial], dim=-1)  # (V, 5)

    # Standardize ch1-ch4 per object (zero-mean / unit-std). ch0 stays in [0,1].
    sub = geom[:, 1:]
    sub = (sub - sub.mean(dim=0, keepdim=True)) / (sub.std(dim=0, keepdim=True) + 1e-6)
    geom = torch.cat([geom[:, :1], sub], dim=-1)
    assert geom.shape == (V, GEOM_DIM)
    return geom.float().contiguous()


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute per-vertex 3D geometry features for the affordance head")
    p.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl")
    p.add_argument("--k", type=int, default=8, help="kNN size; reads vertex_knn_k{K}.pt")
    p.add_argument("--geom_filename", default="vertex_geom.pt", help="output file name per recon dir")
    p.add_argument("--skip_existing", action="store_true", help="skip objects whose geom file already exists")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    data_root = resolve_data_root(cfg)
    rows = load_manifest_rows(args.manifest, data_root=data_root)

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

    knn_name = f"vertex_knn_k{args.k}.pt"
    print(f"{len(recon_dirs)} unique objects in {args.manifest} (data_root={data_root})")
    print(f"Writing {args.geom_filename} (G={GEOM_DIM}, knn={knn_name}) per object …")

    n_done = n_skip = n_miss = 0
    for i, recon_dir in enumerate(recon_dirs):
        out_path = recon_dir / args.geom_filename
        if args.skip_existing and out_path.is_file():
            n_skip += 1
        else:
            pos = _load_vertex_positions(recon_dir)  # centered + unit-sphere normalized; canonical-rotated below
            normals_pt = recon_dir / "vertex_normals.pt"
            knn_pt = recon_dir / knn_name
            if pos is None or not normals_pt.is_file() or not knn_pt.is_file():
                n_miss += 1
                missing = [n for n, ok in [("positions", pos is not None), ("normals", normals_pt.is_file()), (knn_name, knn_pt.is_file())] if not ok]
                print(f"  WARN {recon_dir} missing {missing} — skipped")
            else:
                normals = torch.load(normals_pt, map_location="cpu", weights_only=True).float()
                knn = torch.load(knn_pt, map_location="cpu", weights_only=True).long()
                # Apply the same canonical rotation the dataset applies to positions/normals so the
                # geometry frame (incl. height_up / normal_up) matches the GEAL-labelled orientation.
                canon = recon_dir / "canonical_rotation.pt"
                if canon.is_file():
                    R = torch.load(canon, map_location="cpu", weights_only=True).float()
                    pos = pos.float() @ R.T
                    normals = normals @ R.T
                geom = compute_geom(pos, normals, knn)
                torch.save(geom, out_path)
                n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(recon_dirs)}] computed={n_done} skipped={n_skip} missing={n_miss}")

    print(f"Done: computed={n_done} skipped(existing)={n_skip} missing={n_miss}")


if __name__ == "__main__":
    main()
