"""5-fold CV for AffordanceGNN (EdgeConv message passing over the mesh kNN graph) on human GT.
Direct backbone-vs-backbone test: does SPATIAL REASONING (vertices informed by neighbours) beat the
per-vertex MLP (full-FT+geom, fold0=0.838)? Same inputs (vlm/dino/geom/slat/normals + verb), same
human-GT protocol. Fixed per-object subsample so the rebuilt kNN graph is cached once. Reports per-verb
+ trained-mean. NOTE: GNN trains on human GT only (no GEAL-distillation pretraining the MLP got) — a
positive result is strong; a modest gap is confounded by pretraining (documented)."""
import os, sys, json, argparse, time, collections, hashlib
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, torch
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = str(Path(__file__).resolve().parent)  # absolute: DataRootDataset resolves relative paths against data_root
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp", "press", "lift"]
MLP_FOLD0 = 0.838  # full-FT + geom MLP, fold0 trained-mean (the bar)

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="0")
ap.add_argument("--epochs", type=int, default=18)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--sub", type=int, default=30000)
ap.add_argument("--pos_weight", type=float, default=5.0)
ap.add_argument("--gnn_hidden", type=int, default=128)
ap.add_argument("--gnn_layers", type=int, default=3)
ap.add_argument("--knn_k", type=int, default=8)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--agg", default="max", choices=["max", "attn"])  # EdgeConv max-pool | attention (transformer-style)
ap.add_argument("--edge_geom", action="store_true")  # add geometric edge features (rel pos, dist, normal agreement)
ap.add_argument("--max_objects", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--tag", default="gnn")
ap.add_argument("--init_from", default=None)   # warm-start GNN from a GEAL-pretrained checkpoint
ap.add_argument("--split_file", default=None)  # arbitrary split (e.g. LOCO loco_<cat>.json); overrides --folds
ap.add_argument("--drop", default="")          # ablation: comma list of feature channels to ZERO (clip,dino,geom,slat,normals)
ap.add_argument("--no_verb", action="store_true")  # ablation: feed a CONSTANT (zero) verb embedding -> verb-agnostic model
ap.add_argument("--train_cap", type=int, default=0)  # label-efficiency: subsample train to N objects (seeded); val untouched
ap.add_argument("--dino_file", default="vertex_dino_fine.pt")  # which per-vertex DINO file to load (view-count ablation)
ap.add_argument("--syn_aug", action="store_true")  # train with synonym paraphrases per verb (open-vocab robustness)
ap.add_argument("--syn_aug_rich", action="store_true")  # richer: more synonyms x descriptive templates (held-out TEST words excluded)
args = ap.parse_args()
_DROPMAP = {"clip": "vertex_features", "dino": "dino_vertex_features", "geom": "vertex_geom", "slat": "slat_vertex_features", "normals": "vertex_normals"}
DROP = {_DROPMAP[d] for d in args.drop.split(",") if d in _DROPMAP}
if DROP: print(f"ABLATION: zeroing features {sorted(DROP)}", flush=True)
dev = torch.device(args.device); rng = np.random.default_rng(0)
print(f"device={dev} agg={args.agg} layers={args.gnn_layers} hidden={args.gnn_hidden} k={args.knn_k}", flush=True)

vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0].to(dev) for v in TRAINED}
if args.no_verb:  # verb-agnostic ablation: same constant embedding for every verb -> head cannot route
    _z = torch.zeros_like(next(iter(emb.values()))); emb = {v: _z for v in TRAINED}
    print("ABLATION: no_verb (constant zero verb embedding for all verbs)", flush=True)
vte = torch.stack([emb[v].cpu() for v in TRAINED])  # (N,512) verb_text buffer
# synonym augmentation: multiple paraphrases per verb; sample one per training step (CLIP-text neighbourhood -> region)
_SYN = {"contain":["contain","put into","store","fill"], "sit":["sit","sit on","sit down","seat"],
        "pour":["pour","pour out","pour liquid","empty out"], "move":["move","slide","relocate","shift"],
        "display":["display","show","present","exhibit"], "grasp":["grasp","grip","grab","clutch"],
        "press":["press","tap","push down","press button"], "lift":["lift","pick up","raise","hoist"]}
# richer augmentation: MORE synonyms x descriptive templates. Held-out TEST words (seize/rest on/decant/
# showcase/reposition/click/elevate/keep inside) are deliberately EXCLUDED so the held-out probe stays clean.
_SYN_RICH = {"contain":["contain","put into","store","fill","hold","place inside"],
             "sit":["sit","sit on","sit down","seat","take a seat","be seated"],
             "pour":["pour","pour out","pour liquid","empty out","tip out","dispense"],
             "move":["move","slide","relocate","shift","drag","push along"],
             "display":["display","show","present","exhibit","put on display"],
             "grasp":["grasp","grip","grab","clutch","hold","clasp"],
             "press":["press","tap","push down","press button","depress","push"],
             "lift":["lift","pick up","raise","hoist","lift up","carry"]}
_TPL = ["{w}", "you {w} this object", "the part where you {w}", "an action to {w}"]
if args.syn_aug_rich:
    syn_emb = {v: [vlm.encode_text([t.format(w=s)])[0].to(dev) for s in _SYN_RICH.get(v,[v]) for t in _TPL] for v in TRAINED}
    args.syn_aug = True  # reuse the forward() paraphrase-sampling path
    print(f"SYN_AUG_RICH: {sum(len(v) for v in syn_emb.values())} phrasings (synonyms x templates) across {len(TRAINED)} verbs", flush=True)
elif args.syn_aug:
    syn_emb = {v: [vlm.encode_text([s])[0].to(dev) for s in _SYN.get(v,[v])] for v in TRAINED}
    print(f"SYN_AUG: {sum(len(v) for v in syn_emb.values())} paraphrases across {len(TRAINED)} verbs", flush=True)
else:
    syn_emb = None

cfg = AffordanceGNNConfig(
    vlm_dim=128, dino_vertex_dim=128, geom_dim=5, sam3d_dim=8, normals_dim=3, pos_dim=0,
    verb_dim=128, num_verbs=8, verb_embedding="text", verb_text_dim=512, verb_in_backbone=True,
    dropout=args.dropout, gnn_hidden=args.gnn_hidden, gnn_layers=args.gnn_layers, knn_k=args.knn_k,
    agg=args.agg, edge_geom=args.edge_geom,
)

ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=args.dino_file,
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
obj2row = {}
for i, r in enumerate(ds.rows):
    obj2row.setdefault(os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")), i)

def verbs_of(o):
    return [v for v in TRAINED if os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")]
def human(o, v):
    return torch.as_tensor(np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1), dtype=torch.float32)

# ── fixed per-object subsample + cached kNN over the subsample (built once) ──
sub_cache = {}
def get_sub(o):
    if o not in sub_cache:
        it = ds[obj2row[o]]; V = len(it["vertex_positions"])
        r = np.random.default_rng(int(hashlib.md5(o.encode()).hexdigest()[:8], 16))  # stable across runs (matches eval harness)
        sel = r.choice(V, min(args.sub, V), replace=False)
        sel = torch.as_tensor(np.sort(sel), dtype=torch.long)
        pos = it["vertex_positions"].float()[sel]
        knn = AffordanceGNN._build_knn(pos, args.knn_k).to(dev)  # (S,k) into subsampled set
        # cache the per-vertex feature slices on CPU (moved per use)
        feat = {k: (it.get(k).float()[sel] if it.get(k) is not None else None)
                for k in ["vertex_features", "dino_vertex_features", "slat_vertex_features", "vertex_normals", "vertex_geom"]}
        feat["vertex_positions"] = it["vertex_positions"].float()[sel]
        for k in DROP:                                   # ablation: zero the dropped channel (train + eval)
            if feat.get(k) is not None: feat[k] = torch.zeros_like(feat[k])
        sub_cache[o] = (sel, knn, feat)
    return sub_cache[o]

def forward(model, o, v):
    sel, knn, feat = get_sub(o)
    g = lambda k: (feat[k].to(dev) if feat[k] is not None else None)
    ve = syn_emb[v][rng.integers(len(syn_emb[v]))] if (args.syn_aug and model.training) else emb[v]  # sample paraphrase in train, exact verb at eval
    return model(ve, vlm_features=g("vertex_features"), dino_vertex=g("dino_vertex_features"),
                 slat_vertex=g("slat_vertex_features"), vertex_normals=g("vertex_normals"),
                 vertex_geom=g("vertex_geom"), vertex_positions=g("vertex_positions"), knn_idx=knn), sel

results = []
_folds = ([("x", args.split_file)] if args.split_file
          else [(str(x), f"{SCR}/cv_fold{x}.json") for x in args.folds.split(",")])
for k, _spath in _folds:
    split = json.load(open(_spath))
    train_objs = [o for o in split["train"] if o in obj2row]
    val_objs = [o for o in split["val"] if o in obj2row]
    if args.train_cap and args.train_cap < len(train_objs):  # label-efficiency: seeded train subsample, val UNTOUCHED
        _r = np.random.default_rng(12345)
        train_objs = [train_objs[i] for i in sorted(_r.choice(len(train_objs), args.train_cap, replace=False))]
        print(f"  train_cap={args.train_cap}: using {len(train_objs)} train objs (val kept = {len(val_objs)})", flush=True)
    if args.max_objects:
        train_objs = train_objs[:args.max_objects]; val_objs = val_objs[:max(4, args.max_objects // 3)]
    model = AffordanceGNN(cfg, verb_text_embeddings=vte).to(dev)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=dev, weights_only=False)["model"]
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"  init_from {args.init_from}: loaded (missing={len(miss)} unexpected={len(unexp)})", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr); pw = torch.tensor(args.pos_weight, device=dev)
    pairs = [(o, v) for o in train_objs for v in verbs_of(o)]
    print(f"fold {k}: {len(train_objs)} train obj, {len(pairs)} pairs, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
    best = -1.0; best_line = ""; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); rng.shuffle(pairs); opt.zero_grad(); tot = 0.0
        for i, (o, v) in enumerate(pairs):
            sel, _, _ = get_sub(o); hb = human(o, v)[sel].to(dev).clamp(0, 1)
            logit, _ = forward(model, o, v)
            loss = Fn.binary_cross_entropy_with_logits(logit, hb, pos_weight=pw) / 8
            loss.backward(); tot += loss.item() * 8
            if (i + 1) % 8 == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        # eval each epoch (small val)
        model.eval(); byv = collections.defaultdict(list)
        with torch.no_grad():
            for o in val_objs:
                for v in verbs_of(o):
                    sel, _, _ = get_sub(o); yb = (human(o, v)[sel].numpy() >= 0.5).astype(int)
                    if yb.sum() == 0 or yb.all(): continue
                    p = torch.sigmoid(forward(model, o, v)[0]).float().cpu().numpy()
                    byv[v].append(average_precision_score(yb, p))
        means = {v: float(np.mean(byv[v])) for v in TRAINED if byv[v]}
        tm = float(np.mean(list(means.values()))) if means else 0.0
        line = f"[{'  '.join(f'{v}={means.get(v,0):.3f}' for v in TRAINED)}]"
        if tm > best:
            best = tm; best_line = line
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, f"outputs/gnn_{args.tag}_fold{k}.pt")
        print(f"  fold{k} ep{ep+1}/{args.epochs} loss={tot/max(1,len(pairs)):.3f} tm={tm:.3f} best={best:.3f} {time.time()-t0:.0f}s {line}", flush=True)
    print(f"=== fold {k} BEST trained-mean={best:.3f} (MLP bar {MLP_FOLD0}) d={best-MLP_FOLD0:+.3f} {best_line} ===", flush=True)
    results.append((k, best))
print(f"\n=== GNN CV DONE ({args.agg}) === " + " ".join(f"fold{k}={b:.3f}" for k, b in results), flush=True)
