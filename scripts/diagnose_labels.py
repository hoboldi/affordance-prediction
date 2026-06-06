#!/usr/bin/env python3
"""
Diagnose affordance label quality: unaligned (current) vs normalise+ICP aligned transfer.

Run inside the container::

    PYTHONPATH=src python scripts/diagnose_labels.py \
        --manifest data/manifests/train_sam3d.jsonl --n 8 --export_glb exports/label_debug
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import trimesh

from datasets.data_root_dataset import DataRootDataset, _affordance_labels_from_anno_ply
from reconstruction.affordance_labels import affordance_labels_aligned


def _bbox(p: np.ndarray) -> str:
    lo, hi = p.min(0), p.max(0)
    c = p.mean(0)
    return f"min={lo.round(2)} max={hi.round(2)} radius={np.linalg.norm(p - c, axis=1).max():.2f}"


def _export_glb(verts_n: np.ndarray, faces: np.ndarray | None, labels: np.ndarray,
                anno_aligned: np.ndarray, out: Path) -> None:
    """Aligned frame: positive verts red, negative grey, aligned anno points blue."""
    colors = np.tile(np.array([180, 180, 180, 255], np.uint8), (len(verts_n), 1))
    colors[labels.astype(bool)] = [220, 30, 30, 255]
    scene_parts: list = []
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=verts_n, faces=faces, process=False)
        mesh.visual.vertex_colors = colors
        scene_parts.append(mesh)
    else:
        scene_parts.append(trimesh.PointCloud(verts_n, colors=colors))
    scene_parts.append(trimesh.PointCloud(anno_aligned, colors=np.tile([30, 30, 220, 255], (len(anno_aligned), 1))))
    trimesh.Scene(scene_parts).export(str(out))


def _faces(mesh_glb: Path) -> np.ndarray | None:
    loaded = trimesh.load(str(mesh_glb), process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    else:
        mesh = loaded
    return np.asarray(mesh.faces) if hasattr(mesh, "faces") else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--export_glb", type=Path, default=None)
    args = ap.parse_args()

    manifest = args.manifest.resolve()
    ds = DataRootDataset(
        manifest_path=manifest,
        data_root=manifest.parent,
        load_vertex_labels_eager=False,
        load_vertex_semantics_eager=False,
    )
    print(f"dataset: {len(ds)} rows  |  threshold={args.threshold}\n")

    rates_un, rates_al, residuals = [], [], []
    n = min(args.n, len(ds))
    for i in range(n):
        row = ds.row(i)
        anno = row.vertex_affordance_path
        mesh_glb = row.sam3d_reconstruction_dir / "mesh.glb"
        splat = row.splat_path

        labels_un = _affordance_labels_from_anno_ply(anno, mesh_glb).numpy()
        dbg = affordance_labels_aligned(splat, anno, mesh_glb, threshold=args.threshold, return_debug=True)

        rates_un.append(float(labels_un.mean()))
        rates_al.append(dbg["positive_rate"])
        residuals.append(dbg["icp_mean_residual"])

        print(f"[{i}] {row.sample_id}  verb={row.verb}")
        print(f"     unaligned : positive rate = {labels_un.mean():.3f}")
        print(f"     aligned   : positive rate = {dbg['positive_rate']:.3f}   ICP residual = {dbg['icp_mean_residual']:.3f}")

        if args.export_glb is not None:
            args.export_glb.mkdir(parents=True, exist_ok=True)
            out = args.export_glb / f"{i:02d}_{row.verb}_aligned.glb"
            _export_glb(dbg["verts_n"], _faces(mesh_glb), dbg["labels"], dbg["anno_aligned"], out)
            print(f"     wrote {out}")
        print()

    def _floor(rates: list) -> float:
        p = float(np.clip(np.mean(rates), 1e-6, 1 - 1e-6))
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))

    print(f"unaligned  positive rate mean={np.mean(rates_un):.3f}  base-rate BCE floor={_floor(rates_un):.3f}")
    print(f"aligned    positive rate mean={np.mean(rates_al):.3f}  base-rate BCE floor={_floor(rates_al):.3f}")
    print(f"ICP residual (unit frame): mean={np.mean(residuals):.3f}  max={np.max(residuals):.3f}")
    print("\nLow ICP residual (<~0.05) => good alignment. Eyeball the GLBs: red region should sit under the blue anno points.")


if __name__ == "__main__":
    main()
