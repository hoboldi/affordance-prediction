"""Score an affordance model's per-vertex predictions against HUMAN ground-truth labels.

Per-verb AUPRC vs human labels (human_gt_labels/<object>/vertex_manuallabels_<verb>.pt, soft [V] in
[0,1], V matches the recon mesh). Verbs are split into TRAINED (the model's verb_to_idx) and UNSEEN
(press / lift / open — never trained; reached via open-vocab CLIP-text verb embeddings).

Two prediction sources:
  --source model : the model's sigmoid predictions (open-vocab: unseen verbs CLIP-encoded on the fly).
  --source geal  : the GEAL teacher's soft pseudolabel for that verb (diagnostic — teacher vs human;
                   the ckpt is then used ONLY to derive the dataset channel config).

Usage:
    PYTHONPATH=src python scripts/eval_human_gt.py --ckpt outputs/gnn_geom/best.pt --source model
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from datasets.data_root_dataset import DataRootDataset  # noqa: E402
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg  # noqa: E402
from models.gnn_head import AffordanceGNN, gnn_head_config_from_model_cfg  # noqa: E402

# Verbs the models trained on (the rest — press/lift/open — are open-vocab UNSEEN).
TRAINED_VERBS = {"contain", "grasp", "pour", "display", "move", "sit"}
UNSEEN_VERBS = {"press", "lift", "open"}

DEFAULT_MANIFEST = "/home/datasets/customDatasets/cmr2/manifest.finepatch.jsonl"


def _load_human_label(path: Path) -> torch.Tensor | None:
    try:
        y = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(y, dict):
        for k in ("vertex_affordance", "labels", "y", "features"):
            if k in y and torch.is_tensor(y[k]):
                y = y[k]
                break
        else:
            return None
    return y.float() if torch.is_tensor(y) else None


def _geal_label_for(item: dict, recon_dir: Path, verb: str) -> torch.Tensor | None:
    """The GEAL soft pseudolabel for `verb` on this object: prefer the row's own label, else load."""
    if item.get("verb") == verb and item.get("vertex_affordance") is not None:
        return item["vertex_affordance"].float()
    for name in (f"vertex_pseudolabels_{verb}.pt", f"affordance_{verb}.pt"):
        p = recon_dir / name
        if p.is_file():
            blob = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(blob, dict):
                for k in ("vertex_affordance", "labels", "y"):
                    if k in blob and torch.is_tensor(blob[k]):
                        return blob[k].float()
                return None
            return blob.float()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Score affordance predictions against HUMAN labels (per-verb AUPRC)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="manifest (used only to fetch per-object FEATURES)")
    ap.add_argument("--human_gt_dir", default=str(_REPO_ROOT / "human_gt_labels"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--source", choices=["model", "geal"], default="model")
    ap.add_argument("--max_objects", type=int, default=None, help="cap number of human-GT objects (smoke testing)")
    ap.add_argument("--split", default=None, help="JSON with {'train':[basenames], 'val':[basenames]} to restrict scored objects")
    ap.add_argument("--split_part", choices=["train", "val", "all"], default="all", help="which split list to score (requires --split)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = ck["model_cfg"]
    backbone = mcfg.get("backbone", "mlp")
    if backbone == "gnn":
        cfg = gnn_head_config_from_model_cfg(mcfg)
        model = AffordanceGNN(cfg)
    else:
        cfg = mlp_head_config_from_model_cfg(mcfg)
        model = AffordanceMLP(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    model.to(device)
    v2i = ck["verb_to_idx"]

    _use_dino = int(cfg.dino_vertex_dim) > 0
    _use_geom = int(getattr(cfg, "geom_dim", 0)) > 0
    _knn_k = int(mcfg.get("knn_k", 8) or 8)
    print(f"ckpt={args.ckpt}  backbone={backbone}  source={args.source}  device={device}")
    print(f"  channels: vlm_dim={cfg.vlm_dim} dino={cfg.dino_vertex_dim}({cfg.dino_filename}) "
          f"geom={getattr(cfg,'geom_dim',0)} normals={cfg.normals_dim} pos={cfg.pos_dim} knn_k={_knn_k}")
    print(f"  trained verbs: {sorted(v2i)}")

    ds = DataRootDataset(
        manifest_path=args.manifest,
        load_vertex_labels_eager=True,             # GEAL labels available for --source geal
        load_vertex_semantics_eager=True,          # per-vertex CLIP features
        load_vertex_dino=_use_dino, dino_filename=mcfg.get("dino_filename", "vertex_dino.pt"),
        load_vertex_knn=(backbone == "gnn"), knn_filename=f"vertex_knn_k{_knn_k}.pt",
        load_vertex_geom=_use_geom, geom_filename=mcfg.get("geom_filename", "vertex_geom.pt") or "vertex_geom.pt",
    )

    # object basename -> one row index (features are verb-independent; any row for the object works).
    obj_row: dict[str, int] = {}
    for i, r in enumerate(ds.rows):
        if r.sam3d_reconstruction_dir is None:
            continue
        base = r.sam3d_reconstruction_dir.name
        obj_row.setdefault(base, i)

    # CLIP text encoder for UNSEEN verbs (open-vocab). Built lazily on first need.
    _clip = {}

    def _verb_input(verb: str):
        if verb in v2i:
            return v2i[verb]
        if "enc" not in _clip:
            from vlm.vlm_wrapper import VLMWrapper, VLMConfig
            _clip["enc"] = VLMWrapper(VLMConfig(device="cpu"))
            _clip["cache"] = {}
        if verb not in _clip["cache"]:
            _clip["cache"][verb] = _clip["enc"].encode_text([verb.replace("_", " ")])[0]
        return _clip["cache"][verb].to(device)

    def _f(t):
        return t.float().to(device) if t is not None else None

    @torch.no_grad()
    def _predict(item: dict, verb: str) -> np.ndarray:
        kw = {}
        if backbone == "gnn":
            kn = item.get("vertex_knn")
            kw["knn_idx"] = kn.long().to(device) if kn is not None else None
        logits = model(
            _verb_input(verb),
            slat_vertex=_f(item.get("slat_vertex_features")),
            vlm_features=_f(item.get("vertex_features")),
            dino_cls=_f(item.get("dino_cls")),
            ss_dino_cls=_f(item.get("ss_dino_cls")),
            vertex_normals=_f(item.get("vertex_normals")),
            vertex_positions=_f(item.get("vertex_positions")),
            dino_vertex=_f(item.get("dino_vertex_features")),
            vertex_geom=_f(item.get("vertex_geom")),
            **kw,
        )
        return torch.sigmoid(logits).cpu().numpy()

    human_dir = Path(args.human_gt_dir)
    obj_dirs = sorted(d for d in human_dir.iterdir() if d.is_dir())

    # Optionally restrict to a held-out split list (by object basename).
    if args.split is not None and args.split_part != "all":
        import json as _json
        split = _json.loads(Path(args.split).read_text())
        keep = set(split.get(args.split_part, []))
        obj_dirs = [d for d in obj_dirs if d.name in keep]
        print(f"  split={args.split_part}: restricting to {len(keep)} listed objects ({len(obj_dirs)} present in human_gt_dir)")

    if args.max_objects:
        obj_dirs = obj_dirs[: args.max_objects]

    per_verb_ap: dict[str, list[float]] = collections.defaultdict(list)
    per_verb_std: dict[str, list[float]] = collections.defaultdict(list)
    n_skip_obj = 0
    skipped_objs: list[str] = []
    n_label_files = n_scored = n_skip_label = 0

    for d in obj_dirs:
        base = d.name
        if base not in obj_row:
            n_skip_obj += 1
            skipped_objs.append(base)
            continue
        item = ds[obj_row[base]]
        recon_dir = ds.rows[obj_row[base]].sam3d_reconstruction_dir

        for lf in sorted(d.glob("vertex_manuallabels_*.pt")):
            n_label_files += 1
            verb = lf.name[len("vertex_manuallabels_"):-len(".pt")]
            y = _load_human_label(lf)
            if y is None:
                n_skip_label += 1
                continue
            y_bin = (y >= 0.5).numpy().astype(int)

            if args.source == "model":
                pred = _predict(item, verb)
            else:  # geal
                g = _geal_label_for(item, recon_dir, verb)
                if g is None:
                    n_skip_label += 1
                    continue
                pred = g.numpy()

            if pred.shape[0] != y_bin.shape[0]:
                n_skip_label += 1
                continue
            if y_bin.sum() == 0:
                n_skip_label += 1
                continue

            per_verb_ap[verb].append(float(average_precision_score(y_bin, pred)))
            per_verb_std[verb].append(float(np.std(pred)))
            n_scored += 1

    # ── Report ──────────────────────────────────────────────────────────────
    def _section(title: str, verbs: list[str]) -> tuple[list[float], list[float]]:
        print(f"\n=== {title} (vs HUMAN) ===")
        print(f"  {'verb':<10} {'AUPRC':>7} {'pred_std':>9} {'n_obj':>6}")
        aps, stds = [], []
        for v in verbs:
            if per_verb_ap.get(v):
                ap = float(np.mean(per_verb_ap[v])); sd = float(np.mean(per_verb_std[v])); n = len(per_verb_ap[v])
                print(f"  {v:<10} {ap:>7.3f} {sd:>9.4f} {n:>6d}")
                aps.append(ap); stds.append(sd)
            else:
                print(f"  {v:<10} {'—':>7} {'—':>9} {0:>6d}")
        return aps, stds

    print("\n" + "=" * 56)
    print(f"HUMAN-GT LEADERBOARD   source={args.source}  ckpt={Path(args.ckpt).parent.name}")
    print("=" * 56)
    tr_verbs = sorted(set(per_verb_ap) & TRAINED_VERBS) or sorted(TRAINED_VERBS)
    un_verbs = sorted(set(per_verb_ap) & UNSEEN_VERBS) or sorted(UNSEEN_VERBS)
    tr_aps, _ = _section("TRAINED verbs", tr_verbs)
    un_aps, _ = _section("UNSEEN verbs (open-vocab: press/lift/open)", un_verbs)

    overall = [a for L in per_verb_ap.values() for a in L]
    print("\n=== means ===")
    print(f"  trained-verb mean AUPRC : {np.mean(tr_aps):.3f}" if tr_aps else "  trained-verb mean AUPRC : —")
    print(f"  unseen-verb  mean AUPRC : {np.mean(un_aps):.3f}" if un_aps else "  unseen-verb  mean AUPRC : —")
    print(f"  overall mean AUPRC      : {np.mean(overall):.3f} (over {n_scored} label files)" if overall else "  overall: no scored labels")

    print(f"\nlabel files seen={n_label_files} scored={n_scored} skipped_labels={n_skip_label}")
    print(f"objects: scored={len(obj_dirs)-n_skip_obj}/{len(obj_dirs)}  skipped(not in manifest)={n_skip_obj}")
    if skipped_objs:
        print("  skipped objects:", ", ".join(skipped_objs[:20]) + (" …" if len(skipped_objs) > 20 else ""))


if __name__ == "__main__":
    main()
