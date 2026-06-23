"""GEAL ground truth vs model predictions, per (object, trained-verb).

Columns: GEAL label | v3 (learned) prediction | v4 (open-vocab) prediction. Only trained verbs are
shown (GEAL has no labels for open-vocab synonyms/new actions). Predictions are auto-scaled per cell.

Usage:
    PYTHONPATH=src python scripts/render_gt_comparison.py \
        --v3 outputs/affordance_mlp_v3/epoch_0030.pt \
        --v4 outputs/affordance_mlp_v4_openvocab/epoch_0030.pt \
        --manifest <data_root>/manifest.pseudolabeled.vsem.jsonl \
        --out outputs/renders/gt_comparison.png
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
from sklearn.metrics import average_precision_score  # noqa: E402

from datasets.data_root_dataset import DataRootDataset  # noqa: E402
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg  # noqa: E402

# (category, trained verb) — mix of seen and held-out (vase, laptop) categories.
PAIRS = [("cup", "grasp"), ("cup", "contain"), ("cup", "pour"),
         ("bottle", "grasp"), ("bottle", "pour"), ("bowl", "contain"),
         ("chair", "sit"), ("chair", "move"),
         ("vase", "contain"), ("vase", "pour"), ("laptop", "display")]


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = AffordanceMLP(mlp_head_config_from_model_cfg(ck["model_cfg"]))
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["verb_to_idx"]


def parse_args():
    p = argparse.ArgumentParser(description="GEAL GT vs v3/v4 predictions")
    p.add_argument("--v3", required=True); p.add_argument("--v4", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default=str(_REPO_ROOT / "outputs/renders/gt_comparison.png"))
    p.add_argument("--subsample", type=int, default=13000)
    return p.parse_args()


def main():
    args = parse_args()
    m3, v2i3 = _load(args.v3)
    m4, v2i4 = _load(args.v4)
    ds = DataRootDataset(manifest_path=args.manifest, load_vertex_labels_eager=True, load_vertex_semantics_eager=True)
    idx = {}
    for i, r in enumerate(ds.rows):
        cat = r.sample_id.split("__")[0]
        if (cat, r.verb) in PAIRS and (cat, r.verb) not in idx:
            idx[(cat, r.verb)] = i
    pairs = [p for p in PAIRS if p in idx]

    def _f(t):
        return t.float() if t is not None else None

    def predict(m, v2i, it, verb):
        with torch.no_grad():
            return torch.sigmoid(m(v2i[verb], slat_vertex=_f(it.get("slat_vertex_features")),
                vlm_features=_f(it.get("vertex_features")), dino_cls=_f(it.get("dino_cls")),
                ss_dino_cls=_f(it.get("ss_dino_cls")), vertex_normals=_f(it.get("vertex_normals")),
                vertex_positions=None)).numpy()

    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(9, 2.5 * len(pairs)))
    for ri, (cat, verb) in enumerate(pairs):
        it = ds[idx[(cat, verb)]]
        xyz = it["vertex_positions"].numpy()
        y = it["vertex_affordance"].numpy()
        p3, p4 = predict(m3, v2i3, it, verb), predict(m4, v2i4, it, verb)
        sel = rng.choice(len(xyz), min(args.subsample, len(xyz)), replace=False)
        ap3 = average_precision_score((y >= 0.5).astype(int), p3) if (y >= 0.5).sum() else float("nan")
        ap4 = average_precision_score((y >= 0.5).astype(int), p4) if (y >= 0.5).sum() else float("nan")
        n3, n4 = Path(args.v3).parent.name, Path(args.v4).parent.name
        cells = [("GEAL ground truth", y, 1.0), (f"{n3} (AUPRC {ap3:.2f})", p3, max(0.05, p3[sel].max())),
                 (f"{n4} (AUPRC {ap4:.2f})", p4, max(0.05, p4[sel].max()))]
        for ci, (name, vals, vmax) in enumerate(cells):
            ax = fig.add_subplot(len(pairs), 3, ri * 3 + ci + 1, projection="3d")
            ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=vals[sel], cmap="turbo", s=2, vmin=0.0, vmax=float(vmax))
            h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
            ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[2] - h, mid[2] + h); ax.set_zlim(mid[1] - h, mid[1] + h)
            ax.view_init(elev=14, azim=-60); ax.set_axis_off(); ax.set_title(name, fontsize=8)
            if ci == 0:
                tag = cat + ("*" if cat in ("vase", "laptop") else "")
                ax.text2D(-0.2, 0.5, f"{tag}/{verb}", transform=ax.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
    fig.suptitle("GEAL ground truth vs v3 (learned) vs v4 (open-vocab) — trained verbs   (*=held-out category)", fontsize=11)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}  ({len(pairs)} object/verb rows)")


if __name__ == "__main__":
    main()
