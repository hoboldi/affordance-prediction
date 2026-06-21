"""Generate per-vertex affordance pseudolabels with the frozen GEAL teacher.

For each manifest row pointing at a SAM3D reconstruction, this:

  1. loads ``mesh.glb`` (same options as the training dataset),
  2. surface-samples ``--n_points`` (default 2048, matching GEAL's training distribution),
  3. runs GEAL for the row's ``(object_class, affordance)`` -> per-point scores in [0, 1],
  4. maps scores back to the mesh vertices by nearest sampled point (intra-mesh interpolation in
     the *same* frame -- NOT the cross-modal ICP alignment we removed),
  5. writes ``vertex_pseudolabels.pt`` (shape ``(V,)`` float32) into the reconstruction dir, and
  6. emits an updated manifest with ``vertex_pseudolabel_path`` set per row.

``object_class`` must be a GEAL/3D-AffordanceNet class (see ``labeling.GEAL_CLASSES``); ``affordance``
(the manifest ``verb``) must be in ``labeling.GEAL_AFFORDANCES``. Provide them per-row in the manifest
(``object_class`` field + ``verb``) or globally via ``--object_class`` / ``--affordance``.

Runtime prerequisites (deferred): GEAL clone at ``external/geal``, weights in ``external/geal/ckpt/``,
optional ``Affordance-Question.csv``. See ``labeling.geal_infer`` and the project README.

Usage:
    python scripts/generate_geal_pseudolabels.py \
        --manifest data/omniobject3d/manifest.jsonl \
        --ckpt external/geal/ckpt/piad_seen.pt \
        --question_csv external/geal/ckpt/Affordance-Question.csv \
        --out_manifest data/omniobject3d/manifest_pseudolabeled.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_cwd = Path(__file__).resolve().parent.parent
if (_cwd / "src").is_dir():
    sys.path.insert(0, str(_cwd / "src"))

import numpy as np
import torch

from datasets.data_root_dataset import _parse_row, resolve_data_root
from labeling.geal_infer import GealLabeler
from utils.config import load_config

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

PSEUDOLABEL_FILENAME = "vertex_pseudolabels.pt"


def _load_mesh_vertices_and_sampler(mesh_path: Path):
    """Return (vertices (V,3) float32, trimesh mesh). Same load options as the training dataset."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    return verts, mesh


def _rel_to(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GEAL per-vertex affordance pseudolabels")
    p.add_argument("--manifest", required=True, help="Input manifest.jsonl")
    p.add_argument("--out_manifest", default=None, help="Output manifest with vertex_pseudolabel_path (default: <manifest>.pseudolabeled.jsonl)")
    p.add_argument("--data_root", default=None, help="Dataset root (default: config paths.data_root / $AFFORDANCE_DATA_ROOT)")
    p.add_argument("--ckpt", required=True, help="GEAL checkpoint (.pt), e.g. external/geal/ckpt/piad_seen.pt")
    p.add_argument("--geal_root", default=None, help="GEAL repo clone (default: <repo>/external/geal)")
    p.add_argument("--config", default=None, help="GEAL eval yaml with model_3d block (default: <geal_root>/config/evaluation.yaml)")
    p.add_argument("--question_csv", default=None, help="Affordance-Question.csv for canonical phrasings (optional)")
    p.add_argument("--n_points", type=int, default=2048, help="Surface points fed to GEAL (default: 2048)")
    p.add_argument("--object_class", default=None, help="Override object class for all rows (must be a GEAL class)")
    p.add_argument("--affordance", default=None, help="Override affordance/verb for all rows (must be a GEAL affordance)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--skip_existing", action="store_true", help="Skip rows whose pseudolabel file already exists")
    p.add_argument("--limit", type=int, default=None, help="Process at most N rows (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else resolve_data_root(cfg)
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_manifest = Path(args.out_manifest).expanduser() if args.out_manifest else manifest_path.with_suffix(".pseudolabeled.jsonl")

    raw_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]
    log.info("Loaded %d manifest rows from %s", len(raw_rows), manifest_path)

    labeler = GealLabeler(args.ckpt, config_path=args.config, geal_root=args.geal_root, device=args.device)

    from scipy.spatial import cKDTree

    n_ok, n_skip, n_fail = 0, 0, 0
    for raw in raw_rows:
        row = _parse_row(raw, data_root=data_root)
        sample_id = row.sample_id

        recon_dir = row.sam3d_reconstruction_dir
        mesh_path = (recon_dir / "mesh.glb") if recon_dir is not None else row.mesh_path
        if mesh_path is None or not Path(mesh_path).is_file():
            log.warning("[%s] no mesh found (%s) — skipping", sample_id, mesh_path)
            n_fail += 1
            continue

        out_dir = recon_dir if recon_dir is not None else Path(mesh_path).parent
        out_path = out_dir / PSEUDOLABEL_FILENAME
        if args.skip_existing and out_path.is_file():
            raw["vertex_pseudolabel_path"] = _rel_to(data_root, out_path)
            n_skip += 1
            continue

        object_class = args.object_class or raw.get("object_class")
        affordance = args.affordance or raw.get("verb") or row.verb
        if not object_class:
            log.warning("[%s] missing object_class (set in manifest or --object_class) — skipping", sample_id)
            n_fail += 1
            continue

        try:
            verts, mesh = _load_mesh_vertices_and_sampler(Path(mesh_path))
            n_pts = min(args.n_points, max(len(verts), 1)) if args.n_points else len(verts)
            sampled, _ = mesh.sample(n_pts, return_index=True) if hasattr(mesh, "sample") else (verts, None)
            sampled = np.asarray(sampled, dtype=np.float32)

            scores = labeler.predict(
                sampled, object_class, affordance, question_csv=args.question_csv
            )  # (P,) in [0, 1]

            # Map sampled-point scores onto mesh vertices (nearest in the shared mesh frame).
            _, idx = cKDTree(sampled).query(verts, k=1, workers=-1)
            vert_scores = torch.from_numpy(scores[idx].astype(np.float32))  # (V,)
            torch.save(vert_scores, out_path)

            raw["vertex_pseudolabel_path"] = _rel_to(data_root, out_path)
            n_ok += 1
            log.info(
                "[%s] %s/%s -> %s  (V=%d, pos_rate=%.3f)",
                sample_id, object_class, affordance, out_path.name, len(verts), float((vert_scores > 0.5).float().mean()),
            )
        except Exception as exc:  # noqa: BLE001 — log and continue over the batch
            log.error("[%s] failed: %r", sample_id, exc)
            n_fail += 1

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        for raw in raw_rows:
            f.write(json.dumps(raw) + "\n")

    log.info("Done: %d labeled, %d skipped, %d failed. Wrote %s", n_ok, n_skip, n_fail, out_manifest)


if __name__ == "__main__":
    main()
