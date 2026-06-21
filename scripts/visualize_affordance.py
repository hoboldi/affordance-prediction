"""Visualize affordance head predictions on a mesh.

Loads a trained checkpoint, predicts per-vertex affordance for one sample, and writes:
  * ``<out>/<sid>_pred.glb`` and ``<out>/<sid>_gt.glb`` — vertex-colored meshes (open in any 3D viewer)
  * ``<out>/<sid>.png`` — headless matplotlib point cloud, GT vs predicted side by side

Usage:
    python scripts/visualize_affordance.py --checkpoint outputs/affordance_mlp_full/best.pt --index 0
    python scripts/visualize_affordance.py --checkpoint .../best.pt --sample_id "Seen/val/mug/GS_0003/grasp"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if (_root / "src").is_dir():
    sys.path.insert(0, str(_root / "src"))

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg


def _color(values: np.ndarray) -> np.ndarray:
    """Map [0,1] scores to RdYlGn RGBA uint8 (red=low, green=high)."""
    return (cm.get_cmap("RdYlGn")(np.clip(values, 0, 1))[:, :3] * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize affordance predictions on a mesh")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("/workspace/data/sam3d/manifest.jsonl"))
    p.add_argument("--split", type=str, default="val", help="train / val / test")
    p.add_argument("--index", type=int, default=0, help="Sample index within the (filtered) split")
    p.add_argument("--sample_id", type=str, default=None, help="Exact sample_id (overrides --index)")
    p.add_argument("--categories", nargs="*", default=None, help="Limit to these object categories (e.g. mug)")
    p.add_argument("--verbs", nargs="*", default=None, help="Limit to these actions/verbs (e.g. grasp)")
    p.add_argument("--out", type=Path, default=Path("outputs/affordance_viz"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_points", type=int, default=20000, help="Subsample for the PNG scatter")
    args = p.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # ── Model ───────────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = mlp_head_config_from_model_cfg(ckpt["model_cfg"])
    model = AffordanceMLP(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    verb_to_idx: dict[str, int] = ckpt["verb_to_idx"]

    # ── Sample ──────────────────────────────────────────────────────────────────
    cat_set = set(args.categories) if args.categories else None
    verb_set = set(args.verbs) if args.verbs else None
    _filt = None
    if cat_set is not None or verb_set is not None:
        def _filt(r):
            if cat_set is not None:
                parts = str(r.sample_id).split("/")
                if len(parts) < 3 or parts[2] not in cat_set:
                    return False
            if verb_set is not None and r.verb not in verb_set:
                return False
            return True
    ds = DataRootDataset(manifest_path=args.manifest, split=args.split,
                         load_vertex_labels_eager=True, row_filter=_filt)
    if args.sample_id is not None:
        idx = next((i for i, r in enumerate(ds.rows) if r.sample_id == args.sample_id), None)
        if idx is None:
            raise SystemExit(f"sample_id {args.sample_id!r} not found in split {args.split!r}")
    else:
        idx = args.index
    s = ds[idx]
    sid = s["sample_id"]
    safe = sid.replace("/", "_")

    def _to(t):
        return t.float().to(device) if t is not None else None

    with torch.no_grad():
        logits = model(
            verb_to_idx[s["verb"]],
            slat_vertex=_to(s.get("slat_vertex_features")),
            vlm_features=_to(s.get("vertex_features")),
            dino_cls=_to(s.get("dino_cls")),
            ss_dino_cls=_to(s.get("ss_dino_cls")),
            vertex_normals=_to(s.get("vertex_normals")),
            vertex_positions=_to(s.get("vertex_positions")),
        )
    probs = torch.sigmoid(logits).cpu().numpy()
    gt = s["vertex_affordance"].numpy()

    # Metrics for this sample
    pred = (probs >= 0.5).astype(float)
    tp = ((pred == 1) & (gt == 1)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()
    iou = tp / max(tp + fp + fn, 1)
    try:
        from sklearn.metrics import average_precision_score

        auprc = average_precision_score(gt, probs) if gt.sum() > 0 else float("nan")
    except Exception:
        auprc = float("nan")
    print(f"{sid}  verb={s['verb']}  posrate={gt.mean():.3f}  IoU@0.5={iou:.3f}  AUPRC={auprc:.3f}")

    # ── Colored GLBs ────────────────────────────────────────────────────────────
    recon = s["sam3d_reconstruction_dir"]
    mesh = trimesh.load(str(recon / "mesh.glb"), force="mesh", process=False)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    for tag, vals in [("pred", probs), ("gt", gt)]:
        tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=_color(vals), process=False)
        out_glb = args.out / f"{safe}_{tag}.glb"
        tm.export(str(out_glb))
        print(f"  wrote {out_glb}")

    # ── PNG point cloud (GT vs Pred), a couple of views ──────────────────────────
    V = verts.shape[0]
    sub = np.random.default_rng(0).choice(V, size=min(args.max_points, V), replace=False)
    pv, pp, pg = verts[sub], probs[sub], gt[sub]

    fig = plt.figure(figsize=(12, 6))
    for col, (vals, title) in enumerate([(pg, "Ground truth"), (pp, "Predicted")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.scatter(pv[:, 0], pv[:, 2], pv[:, 1], c=vals, cmap="RdYlGn", s=2, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)
    fig.suptitle(f"{sid}  (verb={s['verb']}, AUPRC={auprc:.3f}, IoU={iou:.3f})")
    out_png = args.out / f"{safe}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
