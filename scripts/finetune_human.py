"""Fine-tune a pretrained affordance ckpt on HUMAN ground-truth labels (per-object held-out split).

Adapts a GEAL-pretrained model to the small set of human-annotated affordance maps
(human_gt_labels/<obj>/vertex_manuallabels_<verb>.pt, soft [V] in [0,1]). Default is HEAD-ONLY
fine-tuning (--freeze_backbone): freeze the whole backbone, train only the verb projection + final
output head — appropriate for the tiny label budget (~120 train objects).

The saved best.pt / last.pt preserve the SAME model_cfg + verb_to_idx as the base ckpt, so
eval_human_gt.py / eval_verb_conditioning.py load them unchanged.

Usage:
    PYTHONPATH=src python scripts/finetune_human.py \
        --ckpt outputs/ov_concat_finepatch/best.pt --split human_split.json \
        --freeze_backbone --output_dir outputs/ft_mlp_s0
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from datasets.data_root_dataset import DataRootDataset  # noqa: E402
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg  # noqa: E402
from models.gnn_head import AffordanceGNN, gnn_head_config_from_model_cfg  # noqa: E402

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


# Verb augmentation: paraphrases per trained verb. During training we condition on a randomly sampled
# paraphrase (encoded via CLIP-text) while keeping the canonical verb's human labels, so the head learns a
# SMOOTH function over verb-embedding space -> a novel verb landing nearby transfers better. Sets are kept
# DISJOINT (no shared word maps to conflicting labels).
VERB_SYN = {
    "contain": ["contain", "store", "fill", "hold things inside"],
    "sit":     ["sit", "sit on", "be seated on", "perch on"],
    "pour":    ["pour", "pour out", "tip out", "empty"],
    "move":    ["move", "push", "drag", "relocate"],
    "display": ["display", "show", "present", "exhibit"],
    "grasp":   ["grasp", "grab", "grip", "pick up"],
}

# Richer verb encoding (--verb_desc): condition on an affordance DESCRIPTION of the verb (CLIP-text) rather
# than the bare word, so a novel verb whose description shares vocabulary with a trained verb lands near it
# (e.g. "lift: grip and raise..." shares "grip" with grasp -> cos 0.80->0.95). Used for BOTH training and
# validation/eval. Novel-verb descriptions are general (describe the action, not the answer region).
VERB_DESC = {
    "contain": "put things inside this object to hold or store them",
    "sit":     "sit down and rest your weight on this object",
    "pour":    "pour liquid out through the opening of this object",
    "move":    "push or grab this object to move it somewhere else",
    "display": "show or present content on the front of this object",
    "grasp":   "grip and hold this object in your hand",
    "lift":    "grip and raise this object up off the surface",
    "open":    "open this object by moving its lid or door",
    "press":   "press down on the surface of this object with a finger",
}


def _load_expand_geom(model, base_sd, insert_at: int, add: int) -> None:
    """Warm-start a model whose per-vertex input grew by `add` geometry channels inserted at `insert_at`
    (after vlm+dino). Expands in_norm.weight/bias and geom_layers.0.weight along the per-vertex axis; the
    new geom slots are zero-init (LayerNorm weight=1, bias=0) so geometry starts inert and is learned during
    (full-)FT. All other params load unchanged. Mirrors the SLAT-encoder input-reshape warm-start."""
    sd = dict(base_sd)

    def insert(w, dim, fill):
        pre = w.narrow(dim, 0, insert_at)
        post = w.narrow(dim, insert_at, w.shape[dim] - insert_at)
        shape = list(w.shape); shape[dim] = add
        return torch.cat([pre, torch.full(shape, fill, dtype=w.dtype), post], dim=dim)

    if "in_norm.weight" in sd:
        sd["in_norm.weight"] = insert(sd["in_norm.weight"], 0, 1.0)
        sd["in_norm.bias"] = insert(sd["in_norm.bias"], 0, 0.0)
    sd["geom_layers.0.weight"] = insert(sd["geom_layers.0.weight"], 1, 0.0)  # [hidden0, per_vertex_dim] -> cols
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys after geom expand: {unexpected}"
    assert not missing, f"missing keys after geom expand: {missing}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a pretrained affordance ckpt on human labels")
    ap.add_argument("--ckpt", required=True, help="pretrained base checkpoint")
    ap.add_argument("--split", required=True, help="human_split.json {'train':[...], 'val':[...]} by object basename")
    ap.add_argument("--human_gt_dir", default=str(_REPO_ROOT / "human_gt_labels"))
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="manifest (features only; verb-independent per object)")
    ap.add_argument("--freeze_backbone", action="store_true", default=True,
                    help="head-only fine-tuning (default ON): train only verb projection + output head")
    ap.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false",
                    help="train ALL params (disable --freeze_backbone)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--pos_weight", type=float, default=5.0)
    ap.add_argument("--max_vertices", type=int, default=0,
                    help="TRAINING-only vertex subsample (0=off). MLP: randomly keep N vertices per object "
                         "per epoch (seeded by epoch+object) for the loss/backward — makes full (no-freeze) "
                         "FT viable on million-vertex meshes. Validation always uses FULL vertices. "
                         "IGNORED for GNN (subsampling breaks the kNN graph).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dino_filename", default=None,
                    help="override the dino feature filename in model_cfg (e.g. inpainted features); "
                         "saved into the ckpt so eval stays consistent")
    ap.add_argument("--exclude_verb", default=None,
                    help="hold this verb OUT of training (leave-one-verb-out CV): its labels are skipped "
                         "so transfer to it can be measured zero-shot via the CLIP-text embedding")
    ap.add_argument("--geom_dim", type=int, default=-1,
                    help="add per-vertex geometry channels (5 = height/concavity/curvature/normal_up/radial). "
                         "If > the ckpt's geom_dim, the input layer is surgically expanded (base channels kept, "
                         "geom slots zero-init) and geometry is learned during (full-)FT. Reads vertex_geom.pt.")
    ap.add_argument("--verb_aug", action="store_true",
                    help="verb augmentation: condition each training pair on a random paraphrase of its verb "
                         "(CLIP-text encoded) to smooth the verb-embedding->affordance map for better novel-verb "
                         "generalization. Labels unchanged; validation uses the canonical verb.")
    ap.add_argument("--verb_mix", type=float, default=0.0,
                    help="verb MIXUP prob [0..1]: with this prob, blend a training pair's verb with ANOTHER verb "
                         "on the SAME object — mix their CLIP-text embeddings (a*v1+(1-a)*v2) AND their soft labels "
                         "— so the head learns the region BETWEEN verbs (widens the capture radius for novel verbs "
                         "that land between trained verbs, e.g. lift~grasp/move). Validation uses canonical verbs.")
    ap.add_argument("--verb_desc", action="store_true",
                    help="richer verb encoding: condition on a CLIP-text-encoded affordance DESCRIPTION of each "
                         "verb (VERB_DESC) instead of the bare word, for both training and eval, so novel verbs "
                         "whose description shares vocabulary with a trained verb land near it.")
    ap.add_argument("--verb_jitter", type=float, default=0.0,
                    help="verb-embedding JITTER: each step, add isotropic Gaussian noise of magnitude "
                         "sigma*||emb|| to the verb's CLIP embedding, so the head learns a smooth neighborhood "
                         "AROUND each trained verb (widens the capture radius in every direction, unlike mixup's "
                         "on-a-line blend). Labels unchanged; validation uses the clean verb.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base model (backbone-detect) ───────────────────────────────────
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = ck["model_cfg"]
    if args.dino_filename:                       # use inpainted dino; record it so eval matches
        mcfg["dino_filename"] = args.dino_filename
    backbone = mcfg.get("backbone", "mlp")
    if backbone == "gnn":
        cfg = gnn_head_config_from_model_cfg(mcfg)
        model = AffordanceGNN(cfg)
    else:
        cfg = mlp_head_config_from_model_cfg(mcfg)
        _old_geom = int(getattr(cfg, "geom_dim", 0))
        _add_geom = args.geom_dim >= 0 and args.geom_dim != _old_geom
        if _add_geom:
            import dataclasses
            cfg = dataclasses.replace(cfg, geom_dim=args.geom_dim)
        model = AffordanceMLP(cfg)
        if _add_geom:
            _load_expand_geom(model, ck["model"], insert_at=int(cfg.vlm_dim + cfg.dino_vertex_dim),
                              add=args.geom_dim - _old_geom)
            mcfg = {**mcfg, "geom_dim": args.geom_dim,
                    "geom_filename": (mcfg.get("geom_filename") or "vertex_geom.pt")}
            print(f"  + geometry channel: geom_dim {_old_geom}->{args.geom_dim} "
                  f"(input expanded at idx {int(cfg.vlm_dim + cfg.dino_vertex_dim)})")
        else:
            model.load_state_dict(ck["model"])
    if backbone == "gnn":
        model.load_state_dict(ck["model"])
    model.to(device)
    v2i = ck["verb_to_idx"]

    # ── Freeze backbone (head-only) ─────────────────────────────────────────
    if backbone == "gnn":
        train_keys = ("verb_proj", "head")
    else:
        train_keys = ("verb_proj", "out")
    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if any(k in name for k in train_keys):
                p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"ckpt={args.ckpt}  backbone={backbone}  freeze_backbone={args.freeze_backbone}  device={device}")
    print(f"  trainable params: {n_train:,} / {n_total:,}  (train keys: {train_keys})")
    if n_train == 0:
        raise SystemExit("No trainable parameters — check freeze logic / train_keys")

    # ── Dataset (ckpt-derived channel flags) ────────────────────────────────
    _use_dino = int(cfg.dino_vertex_dim) > 0
    _use_geom = int(getattr(cfg, "geom_dim", 0)) > 0
    _knn_k = int(mcfg.get("knn_k", 8) or 8)
    ds = DataRootDataset(
        manifest_path=args.manifest,
        load_vertex_labels_eager=True,
        load_vertex_semantics_eager=True,
        load_vertex_dino=_use_dino, dino_filename=mcfg.get("dino_filename", "vertex_dino.pt"),
        load_vertex_knn=(backbone == "gnn"), knn_filename=f"vertex_knn_k{_knn_k}.pt",
        load_vertex_geom=_use_geom, geom_filename=mcfg.get("geom_filename", "vertex_geom.pt") or "vertex_geom.pt",
    )
    obj_row: dict[str, int] = {}
    for i, r in enumerate(ds.rows):
        if r.sam3d_reconstruction_dir is None:
            continue
        obj_row.setdefault(r.sam3d_reconstruction_dir.name, i)

    split = json.loads(Path(args.split).read_text())
    human_dir = Path(args.human_gt_dir)

    # CLIP text encoder for unseen verbs (open-vocab), built lazily.
    _clip: dict = {}

    def _clip_text(s: str):
        if "enc" not in _clip:
            from vlm.vlm_wrapper import VLMWrapper, VLMConfig
            _clip["enc"] = VLMWrapper(VLMConfig(device="cpu"))
            _clip["cache"] = {}
        if s not in _clip["cache"]:
            _clip["cache"][s] = _clip["enc"].encode_text([s])[0]
        return _clip["cache"][s].to(device)

    def _verb_input(verb: str):
        if args.verb_desc:                                   # richer encoding: condition on the description
            return _clip_text(VERB_DESC.get(verb, verb.replace("_", " ")))
        if verb in v2i:
            return v2i[verb]
        return _clip_text(verb.replace("_", " "))

    def _f(t):
        return t.float().to(device) if t is not None else None

    def _forward_logits(item: dict, verb: str, sub: torch.Tensor | None = None, verb_emb=None) -> torch.Tensor:
        """Logits for one object/verb. If `sub` (long vertex indices on `device`) is given, the
        per-vertex inputs are restricted to that subset (MLP-only path; the verb vector is per-object
        and unaffected). GNN is never subsampled (would break the kNN graph). If `verb_emb` (a
        verb_text_dim float tensor) is given, condition on it directly (used for verb-mixup blends)."""
        def _fs(t):
            if t is None:
                return None
            t = t.float().to(device)
            return t[sub] if sub is not None else t

        kw = {}
        if backbone == "gnn":
            kn = item.get("vertex_knn")
            kw["knn_idx"] = kn.long().to(device) if kn is not None else None
        return model(
            verb_emb if verb_emb is not None else _verb_input(verb),
            slat_vertex=_fs(item.get("slat_vertex_features")),
            vlm_features=_fs(item.get("vertex_features")),
            dino_cls=_f(item.get("dino_cls")),          # global (per-object), never subsampled
            ss_dino_cls=_f(item.get("ss_dino_cls")),    # global (per-object), never subsampled
            vertex_normals=_fs(item.get("vertex_normals")),
            vertex_positions=_fs(item.get("vertex_positions")),
            dino_vertex=_fs(item.get("dino_vertex_features")),
            vertex_geom=_fs(item.get("vertex_geom")),
            **kw,
        )

    # ── Build (object, verb, label_path) training/val pairs ──────────────────
    def _pairs(basenames: list[str]) -> list[tuple[str, str, Path]]:
        out = []
        for base in basenames:
            d = human_dir / base
            if base not in obj_row or not d.is_dir():
                continue
            for lf in sorted(d.glob("vertex_manuallabels_*.pt")):
                verb = lf.name[len("vertex_manuallabels_"):-len(".pt")]
                if args.exclude_verb and verb == args.exclude_verb:
                    continue
                out.append((base, verb, lf))
        return out

    train_pairs = _pairs(split["train"])
    val_pairs = _pairs(split["val"])
    # object -> {verb: label_path}, for verb-mixup (needs two verbs' labels on the same object)
    obj_verb_lf: dict[str, dict[str, Path]] = collections.defaultdict(dict)
    for _b, _v, _lf in train_pairs:
        obj_verb_lf[_b][_v] = _lf
    n_train_obj = len({b for b, _, _ in train_pairs})
    n_val_obj = len({b for b, _, _ in val_pairs})
    print(f"  train: {len(train_pairs)} (obj,verb) pairs over {n_train_obj} objects")
    print(f"  val:   {len(val_pairs)} (obj,verb) pairs over {n_val_obj} objects")

    # Training-only vertex subsampling (MLP only; validation always full).
    subsample = args.max_vertices and args.max_vertices > 0
    if subsample and backbone == "gnn":
        print(f"  NOTE: --max_vertices={args.max_vertices} IGNORED for GNN backbone (subsampling breaks the kNN graph); using FULL vertices.")
        subsample = False
    elif subsample:
        print(f"  train subsample: up to {args.max_vertices} vertices/object (seeded by epoch+object); validation uses FULL vertices.")

    bce = lambda logits, tgt: nn.functional.binary_cross_entropy_with_logits(  # noqa: E731
        logits, tgt, pos_weight=torch.tensor(args.pos_weight, device=logits.device))

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 4), gamma=0.5)
    rng = random.Random(args.seed)

    @torch.no_grad()
    def validate() -> tuple[float, dict[str, float]]:
        model.eval()
        per_verb: dict[str, list[float]] = collections.defaultdict(list)
        for base, verb, lf in val_pairs:
            y = _load_human_label(lf)
            if y is None:
                continue
            y_bin = (y >= 0.5).numpy().astype(int)
            if y_bin.sum() == 0:
                continue
            probs = torch.sigmoid(_forward_logits(ds[obj_row[base]], verb)).cpu().numpy()
            if probs.shape[0] != y_bin.shape[0]:
                continue
            per_verb[verb].append(float(average_precision_score(y_bin, probs)))
        per_verb_mean = {v: float(np.mean(a)) for v, a in per_verb.items() if a}
        val_mean = float(np.mean(list(per_verb_mean.values()))) if per_verb_mean else 0.0
        return val_mean, per_verb_mean

    def save(path: Path, epoch: int, val_mean: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "model_cfg": mcfg,            # SAME cfg as base ckpt -> eval scripts load unchanged
                "verb_to_idx": v2i,
                "val_mean_human_auprc": val_mean,
                "finetune": {"base_ckpt": str(args.ckpt), "freeze_backbone": args.freeze_backbone,
                             "lr": args.lr, "pos_weight": args.pos_weight, "seed": args.seed},
            },
            path,
        )

    # ── Train loop ──────────────────────────────────────────────────────────
    best_val, best_epoch, best_per_verb = -1.0, 0, {}
    epochs_no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = list(train_pairs)
        rng.shuffle(order)
        optimizer.zero_grad()
        pending = 0
        losses: list[float] = []
        for base, verb, lf in order:
            y = _load_human_label(lf)
            if y is None:
                continue
            item = ds[obj_row[base]]
            tgt = y.to(device)

            # Training-only subsample (MLP): pick the SAME vertex subset for the inputs and the target.
            # Seeded by epoch+object so each object gets fresh vertices each epoch (variety) while staying
            # reproducible. Validation never subsamples.
            sub = None
            V = tgt.shape[0]
            if subsample and 0 < args.max_vertices < V:
                g = torch.Generator(device="cpu").manual_seed(args.seed * 1_000_003 + ep * 9973 + (hash(base) & 0xFFFFF))
                sub = torch.randperm(V, generator=g)[: args.max_vertices].to(device)

            # Verb MIXUP: with prob --verb_mix, blend this verb with another verb on the SAME object —
            # mix CLIP-text embeddings AND soft labels — so the head learns the region BETWEEN verbs.
            mixed = False
            if args.verb_mix > 0.0 and verb in v2i and rng.random() < args.verb_mix:
                others = [v for v in obj_verb_lf.get(base, {}) if v != verb and v in v2i]
                if others:
                    v2 = rng.choice(others)
                    y2 = _load_human_label(obj_verb_lf[base][v2])
                    if y2 is not None and y2.shape[0] == V:
                        a = rng.uniform(0.15, 0.85)
                        vemb = a * model.verb_text[v2i[verb]] + (1.0 - a) * model.verb_text[v2i[v2]]
                        logits = _forward_logits(item, verb, sub=sub, verb_emb=vemb)
                        tgt = (a * tgt + (1.0 - a) * y2.to(device))
                        tgt = tgt[sub] if sub is not None else tgt
                        mixed = True
            if not mixed and args.verb_jitter > 0.0 and verb in v2i:
                e0 = model.verb_text[v2i[verb]]
                nz = torch.randn_like(e0); nz = nz / (nz.norm() + 1e-8) * e0.norm() * args.verb_jitter
                logits = _forward_logits(item, verb, sub=sub, verb_emb=e0 + nz)
                if sub is not None:
                    tgt = tgt[sub]
                mixed = True
            if not mixed:
                cond_verb = rng.choice(VERB_SYN.get(verb, [verb])) if args.verb_aug else verb
                logits = _forward_logits(item, cond_verb, sub=sub)
                if sub is not None:
                    tgt = tgt[sub]
            if logits.shape[0] != tgt.shape[0]:
                continue
            loss = bce(logits, tgt)
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
            pending += 1
            if pending == args.grad_accum:
                optimizer.step()
                optimizer.zero_grad()
                pending = 0
        if pending > 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        val_mean, per_verb = validate()
        tr_loss = float(np.mean(losses)) if losses else float("nan")
        pv = "  ".join(f"{v}={a:.3f}" for v, a in sorted(per_verb.items()))
        print(f"epoch {ep:>3}/{args.epochs}  train_bce={tr_loss:.4f}  val_mean_AUPRC={val_mean:.4f}  [{pv}]")

        save(out_dir / "last.pt", ep, val_mean)
        if val_mean > best_val:
            best_val, best_epoch, best_per_verb = val_mean, ep, per_verb
            epochs_no_improve = 0
            save(out_dir / "best.pt", ep, val_mean)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stop at epoch {ep} (no val improvement for {args.patience} epochs)")
                break

    pv = "  ".join(f"{v}={a:.3f}" for v, a in sorted(best_per_verb.items()))
    print(f"\nBEST val_mean_AUPRC={best_val:.4f} at epoch {best_epoch}  [{pv}]")
    print(f"saved: {out_dir / 'best.pt'}  (+ last.pt)")


if __name__ == "__main__":
    main()
