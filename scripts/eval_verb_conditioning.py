"""Quantify a trained affordance head's *verb-conditioning* — does the verb change WHERE, not just the level?

For a deterministic sample of objects with >=2 verb-rows, computes per object/verb:
  - per-verb AUPRC (vs binarized GEAL labels) and prediction contrast (std)
  - cross-verb prediction correlation     (lower => verb moves the field, not just its mean)
  - cross-verb top-10% region IoU          (lower => verbs select DISTINCT regions)
Reported as medians over verb-pairs, split out for the "learnable" verbs (contain/pour/sit/move) that
GEAL actually labels densely (grasp/wrap_grasp are teacher-capped on this bowl-heavy set, so they're
reported but not used for the verdict). Prints a machine-readable VERDICT line for autonomous branching.

Usage:
    PYTHONPATH=src python scripts/eval_verb_conditioning.py \
        --ckpt outputs/affordance_mlp_full/best.pt \
        --manifest <data_root>/manifest.pseudolabeled.vsem.jsonl --max_objects 50
"""
from __future__ import annotations

import argparse
import collections
import itertools
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

LEARNABLE = {"contain", "pour", "sit", "move"}  # densely-labeled verbs (grasp/wrap_grasp teacher-capped)


def _topk_mask(p: np.ndarray, frac: float = 0.1) -> np.ndarray:
    k = max(1, int(len(p) * frac))
    m = np.zeros(len(p), bool)
    m[np.argpartition(-p, k - 1)[:k]] = True
    return m


def _predict(model, v2i, it) -> np.ndarray:
    def _f(t):
        return t.float() if t is not None else None

    with torch.no_grad():
        logits = model(
            v2i.get(it["verb"], 0),
            slat_vertex=_f(it.get("slat_vertex_features")),
            vlm_features=_f(it.get("vertex_features")),
            dino_cls=_f(it.get("dino_cls")),
            ss_dino_cls=_f(it.get("ss_dino_cls")),
            vertex_normals=_f(it.get("vertex_normals")),
            vertex_positions=_f(it.get("vertex_positions")),
            dino_vertex=_f(it.get("dino_vertex_features")),
        )
    return torch.sigmoid(logits).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantify verb-conditioning of an affordance head")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max_objects", type=int, default=50, help="deterministic subsample of multi-verb objects")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = AffordanceMLP(mlp_head_config_from_model_cfg(ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    v2i = ck["verb_to_idx"]
    _use_dino = int(ck["model_cfg"].get("dino_vertex_dim", 0)) > 0

    ds = DataRootDataset(manifest_path=args.manifest, load_vertex_labels_eager=True, load_vertex_semantics_eager=True, load_vertex_dino=_use_dino)

    # Group manifest rows by object (reconstruction dir) -> {verb: row_index}
    by_obj: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for i, r in enumerate(ds.rows):
        by_obj[str(r.sam3d_reconstruction_dir)][r.verb] = i
    multi = [od for od, vd in by_obj.items() if len(vd) >= 2]
    multi.sort()  # deterministic
    if args.max_objects and len(multi) > args.max_objects:  # even stride across the sorted list
        idx = np.linspace(0, len(multi) - 1, args.max_objects).round().astype(int)
        multi = [multi[i] for i in sorted(set(idx))]
    print(f"Evaluating {len(multi)} multi-verb objects (of {len(by_obj)} total)")

    per_verb_ap: dict[str, list[float]] = collections.defaultdict(list)
    per_verb_std: dict[str, list[float]] = collections.defaultdict(list)
    corr_all, corr_learn, iou_all, iou_learn = [], [], [], []

    for od in multi:
        preds: dict[str, np.ndarray] = {}
        for verb, idx in by_obj[od].items():
            it = ds[idx]
            p = _predict(model, v2i, it)
            y = it["vertex_affordance"].numpy()
            preds[verb] = p
            per_verb_std[verb].append(float(p.std()))
            if (y >= 0.5).sum() > 0:
                per_verb_ap[verb].append(float(average_precision_score((y >= 0.5).astype(int), p)))
        for va, vb in itertools.combinations(sorted(preds), 2):
            c = float(np.corrcoef(preds[va], preds[vb])[0, 1])
            ma, mb = _topk_mask(preds[va]), _topk_mask(preds[vb])
            iou = float((ma & mb).sum() / max(1, (ma | mb).sum()))
            corr_all.append(c); iou_all.append(iou)
            if va in LEARNABLE and vb in LEARNABLE:
                corr_learn.append(c); iou_learn.append(iou)

    med = lambda x: float(np.median(x)) if x else float("nan")  # noqa: E731
    print("\n=== per-verb ===")
    for v in sorted(set(per_verb_ap) | set(per_verb_std)):
        ap_v = np.mean(per_verb_ap[v]) if per_verb_ap[v] else float("nan")
        print(f"  {v:11s} AUPRC={ap_v:.3f} (n={len(per_verb_ap[v]):3d})  pred_std={np.mean(per_verb_std[v]):.4f}")
    all_std = float(np.mean([s for L in per_verb_std.values() for s in L])) if per_verb_std else float("nan")
    print("\n=== cross-verb (lower = verb changes WHERE) ===")
    print(f"  all pairs       : corr median={med(corr_all):.3f}  topk-IoU median={med(iou_all):.3f}  (n={len(corr_all)})")
    print(f"  learnable pairs : corr median={med(corr_learn):.3f}  topk-IoU median={med(iou_learn):.3f}  (n={len(corr_learn)})")
    print(f"\nVERDICT corr_learn={med(corr_learn):.3f} iou_learn={med(iou_learn):.3f} std={all_std:.4f}")


if __name__ == "__main__":
    main()
