"""Open-vocab affordance showcase: each object shown with the verbs that FIT it (curated, ragged).

Avoids the nonsensical cells of a full object x verb grid (sit/cup). Each object row lists its own
verbs — trained verbs + fitting UNSEEN phrases (the open-vocab payoff) — encoded via CLIP and run
through the open-vocab (``verb_embedding="text"``) head. Edit CURATED below or pass --spec <json>.

Usage:
    PYTHONPATH=src python scripts/render_cross_product.py \
        --ckpt outputs/affordance_mlp_v4_openvocab/last.pt \
        --manifest <data_root>/manifest.pseudolabeled.vsem.jsonl \
        --out outputs/renders/affordance_showcase.png
"""
from __future__ import annotations

import argparse
import json
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
from vlm.vlm_wrapper import VLMWrapper, VLMConfig  # noqa: E402

# Curated per-object triplets demonstrating open-vocab: (phrase, role) where role is
#   "trained"    — a verb the head trained on (reference region)
#   "synonym"    — an UNSEEN paraphrase of the trained verb (should match the trained region)
#   "new action" — an UNSEEN, semantically DIFFERENT affordance that still fits the object
#                  (should localize to a DIFFERENT region). Edit freely.
# Four columns per object:
#   "trained"    — verb trained ON THIS object (reference region)
#   "synonym"    — unseen paraphrase of the trained verb (should match the trained region)
#   "transfer"   — a verb that IS in the training vocab but learned on OTHER object classes, yet suits
#                  this object too (tests transfer of a known action to a new object)
#   "new action" — a verb phrase NEVER seen in training, but suitable for the object (pure open-vocab)
CURATED: dict[str, list[tuple[str, str]]] = {
    "bottle": [("pour", "trained"),    ("pour out", "synonym"),     ("contain", "transfer"), ("drink from", "new action")],
    "bowl":   [("contain", "trained"), ("hold food", "synonym"),    ("grasp", "transfer"),   ("scoop from", "new action")],
    "chair":  [("sit", "trained"),     ("be seated on", "synonym"), ("grasp", "transfer"),   ("lean on", "new action")],
    "cup":    [("grasp", "trained"),   ("grip", "synonym"),         ("move", "transfer"),    ("drink from", "new action")],
    "vase":   [("contain", "trained"), ("fill", "synonym"),         ("grasp", "transfer"),   ("display flowers in", "new action")],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-vocab curated per-object affordance showcase")
    p.add_argument("--ckpt", required=True, help="open-vocab (verb_embedding=text) checkpoint")
    p.add_argument("--manifest", required=True)
    p.add_argument("--spec", default=None, help="JSON {category: [verb phrases]}; default: built-in CURATED")
    p.add_argument("--out", default=str(_REPO_ROOT / "outputs/renders/affordance_showcase.png"))
    p.add_argument("--subsample", type=int, default=12000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec: dict[str, list[str]] = json.loads(Path(args.spec).read_text()) if args.spec else CURATED

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = mlp_head_config_from_model_cfg(ck["model_cfg"])
    if cfg.verb_embedding != "text":
        raise SystemExit(f"--ckpt is verb_embedding={cfg.verb_embedding!r}; need an open-vocab 'text' model")
    model = AffordanceMLP(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    # Encode every unique phrase once.
    phrases = sorted({p for vs in spec.values() for (p, _role) in vs})
    embs = VLMWrapper(VLMConfig(device="cpu")).encode_text([p.replace("_", " ") for p in phrases])
    emb_of = {p: embs[i] for i, p in enumerate(phrases)}

    ds = DataRootDataset(manifest_path=args.manifest, load_vertex_labels_eager=False, load_vertex_semantics_eager=True,
                         load_vertex_dino=int(cfg.dino_vertex_dim) > 0, dino_filename=cfg.dino_filename)
    pick: dict[str, int] = {}
    for i, r in enumerate(ds.rows):
        cat = r.sample_id.split("__")[0]
        if cat in spec and cat not in pick:
            pick[cat] = i
    objects = [c for c in spec if c in pick]
    if not objects:
        raise SystemExit("No spec objects found in manifest.")

    def _f(t):
        return t.float() if t is not None else None

    ncol = max(len(spec[c]) for c in objects)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(2.5 * ncol, 2.7 * len(objects)))
    for ri, cat in enumerate(objects):
        it = ds[pick[cat]]
        xyz = it["vertex_positions"].numpy()
        sel = rng.choice(len(xyz), min(args.subsample, len(xyz)), replace=False)
        kw = dict(slat_vertex=_f(it.get("slat_vertex_features")), vlm_features=_f(it.get("vertex_features")),
                  dino_cls=_f(it.get("dino_cls")), ss_dino_cls=_f(it.get("ss_dino_cls")),
                  vertex_normals=_f(it.get("vertex_normals")), vertex_positions=None,
                  dino_vertex=_f(it.get("dino_vertex_features")))
        for ci, (verb, role) in enumerate(spec[cat]):
            with torch.no_grad():
                p = torch.sigmoid(model(emb_of[verb], **kw)).numpy()
            ax = fig.add_subplot(len(objects), ncol, ri * ncol + ci + 1, projection="3d")
            ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=p[sel], cmap="turbo", s=2,
                       vmin=0.0, vmax=float(max(0.05, p[sel].max())))
            h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6
            mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
            ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[2] - h, mid[2] + h); ax.set_zlim(mid[1] - h, mid[1] + h)
            ax.view_init(elev=14, azim=-60); ax.set_axis_off()
            colr = {"trained": "black", "synonym": "tab:green", "transfer": "tab:blue", "new action": "tab:red"}.get(role, "black")
            ax.set_title(f"{verb}\n({role})", fontsize=8, color=colr)
            if ci == 0:
                ax.text2D(-0.18, 0.5, cat, transform=ax.transAxes, fontsize=11, fontweight="bold", rotation=90, va="center")
    fig.suptitle("Open-vocab affordance (v15) — trained (black) | synonym (green) | transfer: known verb, new object (blue) | never-seen (red)", fontsize=10)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}  ({len(objects)} objects, phrases: {phrases})")


if __name__ == "__main__":
    main()
