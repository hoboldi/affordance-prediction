"""Render per-vertex affordance fields on SAM3D reconstructions to a PNG (headless, matplotlib).

Shows the GEAL pseudolabel (teacher) and, with ``--ckpt``, the trained head's prediction (student)
side by side for each object, using the same V-aligned ``vertex_positions`` the head trains on
(no OpenGL / pyrender needed). Useful as a qualitative sanity check on labels + predictions.

Usage:
    PYTHONPATH=src python scripts/render_affordance.py \
        --manifest <data_root>/manifest.pseudolabeled.jsonl \
        --ckpt outputs/affordance_mlp_smoke/best.pt \
        --sample_ids chair__555_79602_154353__v000 bowl__69_5465_12831__v000 vase__380_44863_89631__v000 \
        --out outputs/renders/affordance_smoke.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from datasets.data_root_dataset import DataRootDataset  # noqa: E402
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render affordance fields on reconstructions")
    p.add_argument("--manifest", required=True, help="Pseudolabeled manifest.jsonl")
    p.add_argument("--ckpt", default=None, help="Trained head checkpoint (adds prediction column)")
    p.add_argument("--sample_ids", nargs="*", default=None, help="Specific sample_ids (default: first --limit)")
    p.add_argument("--limit", type=int, default=3, help="Number of objects if --sample_ids not given")
    p.add_argument("--subsample", type=int, default=20000, help="Vertices to scatter per object")
    p.add_argument("--out", default=str(_REPO_ROOT / "outputs/renders/affordance.png"))
    return p.parse_args()


def _load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AffordanceMLP(mlp_head_config_from_model_cfg(ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["verb_to_idx"]


def _predict(model, verb_to_idx, item) -> np.ndarray:
    def _f(t):
        return t.float() if t is not None else None

    with torch.no_grad():
        logits = model(
            verb_to_idx.get(item["verb"], 0),
            slat_vertex=_f(item.get("slat_vertex_features")),
            vlm_features=None,
            dino_cls=_f(item.get("dino_cls")),
            ss_dino_cls=_f(item.get("ss_dino_cls")),
            vertex_normals=_f(item.get("vertex_normals")),
            vertex_positions=_f(item.get("vertex_positions")),
        )
    return torch.sigmoid(logits).numpy()


def _scatter(ax, xyz: np.ndarray, vals: np.ndarray, title: str):
    # Render +Y-up: map data (x,y,z) -> plot (x,z,y) so the SAM3D vertical axis (+Y) is vertical on screen.
    p = ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=vals, cmap="turbo", s=2, vmin=0.0, vmax=1.0)
    half = float((xyz.max(0) - xyz.min(0)).max()) / 2 + 1e-6
    mid = (xyz.max(0) + xyz.min(0)) / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[2] - half, mid[2] + half)
    ax.set_zlim(mid[1] - half, mid[1] + half)
    ax.view_init(elev=15, azim=-60)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    return p


def main() -> None:
    args = parse_args()
    ds = DataRootDataset(
        manifest_path=args.manifest,
        load_mesh_eager=False,
        load_vertex_labels_eager=True,
        load_vertex_semantics_eager=False,
    )
    rows = ds.rows
    if args.sample_ids:
        picks = [i for i, r in enumerate(rows) if r.sample_id in set(args.sample_ids)]
    else:
        picks = list(range(min(args.limit, len(rows))))
    if not picks:
        raise SystemExit("No matching samples in manifest.")

    model = verb_to_idx = None
    if args.ckpt:
        model, verb_to_idx = _load_model(args.ckpt)

    ncol = 2 if model is not None else 1
    fig = plt.figure(figsize=(5.2 * ncol, 4.2 * len(picks)))
    rng = np.random.default_rng(0)
    last = None

    for r, idx in enumerate(picks):
        item = ds[idx]
        xyz = item.get("vertex_positions")
        y = item.get("vertex_affordance")
        if xyz is None or y is None:
            continue
        xyz = xyz.numpy()
        y = y.numpy()
        obj = item["extras"].get("object_class", "")
        n = len(xyz)
        sel = rng.choice(n, min(args.subsample, n), replace=False)

        panels = [(f"{item['sample_id'][:26]}\n{obj} / {item['verb']} — GEAL label  (pos={float((y>=0.5).mean()):.2f})", y)]
        if model is not None:
            pred = _predict(model, verb_to_idx, item)
            panels.append((f"{obj} / {item['verb']} — head prediction", pred))

        for c, (title, vals) in enumerate(panels):
            ax = fig.add_subplot(len(picks), ncol, r * ncol + c + 1, projection="3d")
            last = _scatter(ax, xyz[sel], vals[sel], title)

    if last is not None:
        fig.colorbar(last, ax=fig.axes, shrink=0.5, label="affordance score [0,1]")
    fig.suptitle("Affordance fields on SAM3D reconstructions (real CO3D images)", fontsize=11)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
