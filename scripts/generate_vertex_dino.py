"""Per-vertex DINOv2 patch features for SAM3D reconstructions (render -> DINOv2 patches -> project).

Mirrors generate_vertex_semantics.py but uses DINOv2 (facebook/dinov2-base) instead of CLIP. DINOv2 patch
tokens are more part-aware / localization-friendly than CLIP, so this gives the affordance head a sharper
per-vertex appearance channel for the affordance head. Writes ``vertex_dino.pt`` (dict: features (V,D) fp16,
visible_in_any_view (V,) bool) per recon dir.

Usage:
    PYTHONPATH=src python scripts/generate_vertex_dino.py \
        --manifest <data_root>/manifest.pseudolabeled.vsem.jsonl --reduce_dim 128 --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch
import trimesh
from PIL import Image

from datasets.data_root_dataset import _parse_row, resolve_data_root
from datasets.mesh_loading import load_mesh
from projection.project_to_mesh import ViewProjectionInputs, project_views_to_vertices
from rendering.renderer import render_mesh_views
from utils.config import load_config
from vlm.patch_extractor import PatchFeatures

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

DINO_FILENAME = "vertex_dino.pt"
DINO_MODEL = "facebook/dinov2-base"  # 768-dim patch tokens, patch_size 14


class DinoExtractor:
    def __init__(self, device: str | None = None, model_name: str = DINO_MODEL, image_size: int | None = None):
        from transformers import AutoImageProcessor, AutoModel
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size  # finer patch grid: e.g. 448 -> 32x32 (vs default 224 -> 16x16)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if image_size is not None:  # we resize ourselves; stop the processor from resizing/cropping back to 224
            self.processor.do_resize = False
            if hasattr(self.processor, "do_center_crop"):
                self.processor.do_center_crop = False
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim = int(self.model.config.hidden_size)
        # Determine the patch grid from an actual forward (robust to processor config / register tokens).
        with torch.inference_mode():
            inp = self._proc(np.zeros(((image_size or 64), (image_size or 64), 3), np.uint8))
            n_tok = self.model(**inp).last_hidden_state.shape[1] - 1  # minus CLS
        self.grid = int(round(n_tok ** 0.5))

    def _proc(self, img: np.ndarray):
        pil = Image.fromarray(img.astype(np.uint8))
        if self.image_size is not None:
            pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        return self.processor(images=pil, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def patches(self, images: list[np.ndarray]) -> list[PatchFeatures]:
        out: list[PatchFeatures] = []
        for img in images:
            hidden = self.model(**self._proc(img)).last_hidden_state  # (1, 1+N, D)
            patch = hidden[0, 1:, :].cpu().float()        # drop CLS -> (N, D)
            out.append(PatchFeatures(patches=patch, grid_h=self.grid, grid_w=self.grid, feature_dim=patch.shape[-1]))
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-vertex DINOv2 features")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_manifest", default=None, help="default: <manifest>.dino.jsonl")
    p.add_argument("--data_root", default=None)
    p.add_argument("--reduce_dim", type=int, default=128, help="random-projection target dim (0 = keep full)")
    p.add_argument("--num_views", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dino_model", default=DINO_MODEL, help="HF DINOv2 model (e.g. facebook/dinov2-large, facebook/dinov2-with-registers-large)")
    p.add_argument("--dino_filename", default=DINO_FILENAME, help="output filename per recon dir (use a distinct name to coexist with the default)")
    p.add_argument("--dino_image_size", type=int, default=None, help="DINO input size; 448 -> 32x32 patches (finer localization) vs default 224 -> 16x16")
    p.add_argument("--elevation_rings", default=None, help="comma-sep elevations (deg) overriding cfg, e.g. '43' or '33,52' (view ablation)")
    p.add_argument("--azimuths_per_ring", type=int, default=None, help="azimuths per ring overriding cfg (view ablation)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg.setdefault("rendering", {})["backend"] = "mesh"
    if args.num_views is not None:
        cfg["rendering"]["num_views"] = args.num_views
    if args.elevation_rings is not None:
        cfg["rendering"]["elevation_rings_deg"] = [float(x) for x in args.elevation_rings.split(",")]
    if args.azimuths_per_ring is not None:
        cfg["rendering"]["azimuths_per_ring"] = args.azimuths_per_ring
    clip_image_size = int(cfg.get("projection", {}).get("clip_image_size", 224))

    data_root = Path(args.data_root).resolve() if args.data_root else resolve_data_root(cfg)
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_manifest = Path(args.out_manifest).expanduser() if args.out_manifest else manifest_path.with_suffix(".dino.jsonl")
    raw_rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]

    recon_dirs: dict[str, Path] = {}
    for raw in raw_rows:
        row = _parse_row(raw, data_root=data_root)
        if row.sam3d_reconstruction_dir is not None:
            recon_dirs.setdefault(str(row.sam3d_reconstruction_dir), row.sam3d_reconstruction_dir)
    dirs = list(recon_dirs.values())
    if args.limit:
        dirs = dirs[: args.limit]

    extractor = DinoExtractor(args.device, args.dino_model, image_size=args.dino_image_size)
    log.info("DINOv2 %s (dim=%d, grid=%dx%d) -> %s reduce=%d for %d objects", args.dino_model, extractor.dim, extractor.grid, extractor.grid, args.dino_filename, args.reduce_dim, len(dirs))

    rng = np.random.default_rng(0)
    proj = None
    if args.reduce_dim and args.reduce_dim > 0:
        proj = (rng.standard_normal((extractor.dim, args.reduce_dim)) / np.sqrt(args.reduce_dim)).astype(np.float32)

    n_ok = n_skip = n_fail = 0
    for d in dirs:
        out_path = d / args.dino_filename
        if args.skip_existing and out_path.is_file():
            n_skip += 1
            continue
        mesh_path = d / "mesh.glb"
        if not mesh_path.is_file():
            log.warning("[%s] no mesh.glb", d.name); n_fail += 1; continue
        try:
            mesh = load_mesh(mesh_path, process=False)
            views = render_mesh_views(mesh, cfg)
            patches = extractor.patches([v.rgb for v in views])
            vnormals = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).vertex_normals
            pv = [ViewProjectionInputs(vertex_uv=v.vertex_uv, vertex_visible=v.vertex_visible, patches=p,
                                       render_height=v.rgb.shape[0], render_width=v.rgb.shape[1],
                                       camera_position=v.camera_pose[:3, 3])
                  for v, p in zip(views, patches)]
            vs = project_views_to_vertices(pv, clip_image_size=clip_image_size,
                                           vertex_normals=vnormals, vertex_positions=mesh.vertices)
            feats = vs.features.float()
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
            if proj is not None:
                feats = feats @ torch.from_numpy(proj)
            torch.save({"features": feats.half(),
                        "visible_in_any_view": torch.from_numpy(np.asarray(vs.visible_in_any_view, bool))}, out_path)
            n_ok += 1
            log.info("[%s] dino %s visible=%.2f", d.name, tuple(feats.shape), float(vs.visible_in_any_view.mean()))
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] failed: %r", d.name, exc); n_fail += 1

    for raw in raw_rows:
        row = _parse_row(raw, data_root=data_root)
        if row.sam3d_reconstruction_dir is not None:
            vp = row.sam3d_reconstruction_dir / args.dino_filename
            if vp.is_file():
                try:
                    raw["vertex_dino_path"] = str(vp.resolve().relative_to(data_root.resolve()))
                except ValueError:
                    raw["vertex_dino_path"] = str(vp.resolve())
    out_manifest.write_text("\n".join(json.dumps(r) for r in raw_rows) + "\n")
    log.info("Done: %d ok, %d skipped, %d failed. Wrote %s", n_ok, n_skip, n_fail, out_manifest)


if __name__ == "__main__":
    main()
