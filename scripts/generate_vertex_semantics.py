"""Per-vertex CLIP semantic features for SAM3D reconstructions (render -> CLIP patches -> project).

For each object (reconstruction dir) in the manifest: render the mesh from several views, extract frozen
CLIP patch features per view, project + fuse them onto the mesh vertices, then L2-normalize and reduce
512 -> ``--reduce_dim`` via a FIXED random (Johnson-Lindenstrauss) projection, stored float16. Writes
``vertex_semantics.pt`` (dict: ``features`` (V,D) fp16, ``visible_in_any_view`` (V,) bool) into each recon
dir and emits a manifest with ``vertex_semantics_path`` set per row.

These per-vertex features give the affordance head the spatial+semantic signal that SLAT-8 + normals
lack, so the verb-conditioned head can localize different regions per verb (set model.vlm_dim = D).

Usage:
    PYTHONPATH=src python scripts/generate_vertex_semantics.py \
        --manifest data/.../manifest.pseudolabeled.jsonl --reduce_dim 128 --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # headless rendering (NVIDIA EGL)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch
import trimesh

from datasets.data_root_dataset import _parse_row, resolve_data_root
from datasets.mesh_loading import load_mesh
from projection.project_to_mesh import ViewProjectionInputs, project_views_to_vertices
from rendering.renderer import render_mesh_views
from utils.config import load_config
from vlm.patch_extractor import extract_patch_features
from vlm.vlm_wrapper import VLMWrapper, build_vlm_config

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

VSEM_FILENAME = "vertex_semantics.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-vertex CLIP semantic features")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_manifest", default=None, help="default: <manifest>.vsem.jsonl")
    p.add_argument("--data_root", default=None)
    p.add_argument("--reduce_dim", type=int, default=128, help="random-projection target dim (0 = keep full 512)")
    p.add_argument("--num_views", type=int, default=None, help="override rendering.num_views")
    p.add_argument("--device", default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="process at most N objects (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.setdefault("rendering", {})["backend"] = "mesh"
    if args.num_views is not None:
        cfg["rendering"]["num_views"] = args.num_views
    if args.device:
        cfg.setdefault("vlm", {})["device"] = args.device
    clip_image_size = int(cfg.get("projection", {}).get("clip_image_size", 224))

    data_root = Path(args.data_root).resolve() if args.data_root else resolve_data_root(cfg)
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_manifest = Path(args.out_manifest).expanduser() if args.out_manifest else manifest_path.with_suffix(".vsem.jsonl")

    raw_rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]

    # Unique reconstruction dirs (one object may have several verb rows).
    recon_dirs: dict[str, Path] = {}
    for raw in raw_rows:
        row = _parse_row(raw, data_root=data_root)
        if row.sam3d_reconstruction_dir is not None:
            recon_dirs.setdefault(str(row.sam3d_reconstruction_dir), row.sam3d_reconstruction_dir)
    dirs = list(recon_dirs.values())
    if args.limit:
        dirs = dirs[: args.limit]
    log.info("Computing vertex semantics for %d unique objects (%d manifest rows)", len(dirs), len(raw_rows))

    # Fixed random projection 512 -> reduce_dim (deterministic; comparable across objects).
    rng = np.random.default_rng(0)
    proj = None
    if args.reduce_dim and args.reduce_dim > 0:
        proj = (rng.standard_normal((512, args.reduce_dim)) / np.sqrt(args.reduce_dim)).astype(np.float32)

    wrapper = VLMWrapper(build_vlm_config(cfg))
    n_ok = n_skip = n_fail = 0
    for d in dirs:
        vsem_path = d / VSEM_FILENAME
        if args.skip_existing and vsem_path.is_file():
            n_skip += 1
            continue
        mesh_path = d / "mesh.glb"
        if not mesh_path.is_file():
            log.warning("[%s] no mesh.glb", d.name); n_fail += 1; continue
        try:
            mesh = load_mesh(mesh_path, process=False)  # process=False => V aligned to slat/labels
            views = render_mesh_views(mesh, cfg)
            patches = extract_patch_features(wrapper, [v.rgb for v in views])
            vnormals = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).vertex_normals
            pv = [
                ViewProjectionInputs(
                    vertex_uv=v.vertex_uv, vertex_visible=v.vertex_visible, patches=p,
                    render_height=v.rgb.shape[0], render_width=v.rgb.shape[1],
                    camera_position=v.camera_pose[:3, 3],
                )
                for v, p in zip(views, patches)
            ]
            vs = project_views_to_vertices(pv, clip_image_size=clip_image_size,
                                           vertex_normals=vnormals, vertex_positions=mesh.vertices)
            feats = vs.features.float()                                   # (V, 512)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)     # L2-normalize per vertex
            if proj is not None:
                feats = feats @ torch.from_numpy(proj)                   # (V, reduce_dim)
            torch.save(
                {  # tensors only, so the data loader can torch.load(weights_only=True)
                    "features": feats.half(),
                    "visible_in_any_view": torch.from_numpy(np.asarray(vs.visible_in_any_view, bool)),
                },
                vsem_path,
            )
            n_ok += 1
            log.info("[%s] vsem %s  visible=%.2f", d.name, tuple(feats.shape), float(vs.visible_in_any_view.mean()))
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] failed: %r", d.name, exc); n_fail += 1

    # Emit manifest with vertex_semantics_path per row (keyed on the row's reconstruction dir).
    for raw in raw_rows:
        row = _parse_row(raw, data_root=data_root)
        if row.sam3d_reconstruction_dir is not None:
            vp = row.sam3d_reconstruction_dir / VSEM_FILENAME
            if vp.is_file():
                try:
                    raw["vertex_semantics_path"] = str(vp.resolve().relative_to(data_root.resolve()))
                except ValueError:
                    raw["vertex_semantics_path"] = str(vp.resolve())
    out_manifest.write_text("\n".join(json.dumps(r) for r in raw_rows) + "\n")
    log.info("Done: %d ok, %d skipped, %d failed. Wrote %s", n_ok, n_skip, n_fail, out_manifest)


if __name__ == "__main__":
    main()
