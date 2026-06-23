"""Generalization eval: per-category AUPRC (seen vs held-out) + open-vocab synonym-consistency.

1. Per-category AUPRC — groups eval objects by category/verb, so we can read unseen-category
   generalization (vase, laptop are held out of training) vs seen categories.
2. Synonym-consistency (only for verb_embedding="text" checkpoints) — for (trained, synonym, unrelated)
   triples, corr(pred(synonym), pred(trained)) should be HIGH and corr(pred(unrelated), pred(trained))
   LOW. The gap is a label-free measure of open-vocab phrasing-invariance.

Usage:
    PYTHONPATH=src python scripts/eval_generalization.py --ckpt <ckpt> \
        --manifest <data_root>/manifest.pseudolabeled.vsem.jsonl --max_per_cat 25
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

HELD_OUT = {"vase", "laptop"}
# (trained verb, unseen synonym, unrelated trained verb)
SYN_TRIPLES = [("grasp", "grip", "contain"), ("contain", "fill", "grasp"),
               ("pour", "pour out", "sit"), ("sit", "be seated on", "contain")]


def _f(t):
    return t.float() if t is not None else None


def _predict(model, verb, it):
    with torch.no_grad():
        return torch.sigmoid(model(verb, slat_vertex=_f(it.get("slat_vertex_features")),
            vlm_features=_f(it.get("vertex_features")), dino_cls=_f(it.get("dino_cls")),
            ss_dino_cls=_f(it.get("ss_dino_cls")), vertex_normals=_f(it.get("vertex_normals")),
            vertex_positions=None, dino_vertex=_f(it.get("dino_vertex_features")))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max_per_cat", type=int, default=25)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = mlp_head_config_from_model_cfg(ck["model_cfg"])
    model = AffordanceMLP(cfg); model.load_state_dict(ck["model"]); model.eval()
    v2i = ck["verb_to_idx"]
    ds = DataRootDataset(manifest_path=args.manifest, load_vertex_labels_eager=True, load_vertex_semantics_eager=True, load_vertex_dino=int(cfg.dino_vertex_dim) > 0, dino_filename=cfg.dino_filename)

    # 1. per-category AUPRC (capped per category/verb for speed)
    seen_cnt: dict = collections.defaultdict(int)
    by_cat = collections.defaultdict(list)
    by_cat_verb = collections.defaultdict(list)
    for i, r in enumerate(ds.rows):
        cat = r.sample_id.split("__")[0]
        if seen_cnt[(cat, r.verb)] >= args.max_per_cat:
            continue
        it = ds[i]
        y = it["vertex_affordance"].numpy()
        if (y >= 0.5).sum() == 0:
            continue
        ap_v = float(average_precision_score((y >= 0.5).astype(int), _predict(model, v2i.get(r.verb, 0), it)))
        seen_cnt[(cat, r.verb)] += 1
        by_cat[cat].append(ap_v)
        by_cat_verb[(cat, r.verb)].append(ap_v)
    print("=== per-category AUPRC (vs GEAL)  [*=held-out category] ===")
    for cat in sorted(by_cat):
        tag = cat + ("*" if cat in HELD_OUT else "")
        verbs = ", ".join(f"{vv}={np.mean(by_cat_verb[(cc, vv)]):.2f}" for (cc, vv) in sorted(by_cat_verb) if cc == cat)
        print(f"  {tag:10s} mean={np.mean(by_cat[cat]):.3f} (n={len(by_cat[cat])})   {verbs}")
    seen = [a for c in by_cat for a in by_cat[c] if c not in HELD_OUT]
    held = [a for c in by_cat for a in by_cat[c] if c in HELD_OUT]
    print(f"  --> SEEN categories mean AUPRC={np.mean(seen):.3f} | HELD-OUT (vase,laptop) mean={np.mean(held) if held else float('nan'):.3f}")

    # 2. synonym-consistency (open-vocab only)
    if cfg.verb_embedding != "text":
        print("\n(skip synonym-consistency: not an open-vocab 'text' checkpoint)")
        return
    from vlm.vlm_wrapper import VLMWrapper, VLMConfig
    wrap = VLMWrapper(VLMConfig(device="cpu"))
    phrases = sorted({p for t in SYN_TRIPLES for p in t})
    emb = {p: e for p, e in zip(phrases, wrap.encode_text([p.replace("_", " ") for p in phrases]))}
    syn_c, unrel_c = collections.defaultdict(list), collections.defaultdict(list)
    for i, r in enumerate(ds.rows):
        for trained, syn, unrel in SYN_TRIPLES:
            if r.verb != trained:
                continue
            it = ds[i]
            pt = _predict(model, v2i.get(trained, 0), it)
            ps = _predict(model, emb[syn], it)
            pu = _predict(model, emb[unrel], it)
            syn_c[trained].append(float(np.corrcoef(pt, ps)[0, 1]))
            unrel_c[trained].append(float(np.corrcoef(pt, pu)[0, 1]))
    print("\n=== open-vocab synonym-consistency (corr with the trained verb's prediction) ===")
    for trained, syn, unrel in SYN_TRIPLES:
        if syn_c[trained]:
            print(f"  {trained:9s}: synonym '{syn}' corr={np.mean(syn_c[trained]):+.3f}  vs  unrelated '{unrel}' corr={np.mean(unrel_c[trained]):+.3f}  (n={len(syn_c[trained])})")
    alls = [c for v in syn_c.values() for c in v]; allu = [c for v in unrel_c.values() for c in v]
    print(f"  --> mean synonym corr={np.mean(alls):+.3f}  unrelated corr={np.mean(allu):+.3f}  GAP={np.mean(alls)-np.mean(allu):+.3f} (higher = better open-vocab)")


if __name__ == "__main__":
    main()
