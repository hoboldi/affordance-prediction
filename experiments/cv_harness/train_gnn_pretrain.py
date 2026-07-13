"""Pretrain AffordanceGNN on GEAL pseudolabels (1227 objects, 6 verbs) — mirrors how the MLP base was
made, then human-FT warm-starts from this. Memory-safe: LAZY dataset (no eager preload); on first touch
per object, subsample to --sub verts + build kNN + cache features in fp16 (~21 GB for the full set), reused
across epochs. Saves {model, cfg} to --out for train_gnn_cv.py --init_from. GPU."""
import os, sys, json, argparse, time
sys.path.insert(0, "src")
import numpy as np, torch
import torch.nn.functional as Fn
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp", "press", "lift"]
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{ROOT}/manifest.pseudolabeled.clean.dino.jsonl")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--sub", type=int, default=30000)
ap.add_argument("--pos_weight", type=float, default=5.0)
ap.add_argument("--gnn_hidden", type=int, default=128)
ap.add_argument("--gnn_layers", type=int, default=3)
ap.add_argument("--knn_k", type=int, default=24)
ap.add_argument("--max_objects", type=int, default=0)   # smoke
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="outputs/gnn_pretrain.pt")
args = ap.parse_args()
dev = torch.device(args.device); rng = np.random.default_rng(0)
print(f"device={dev} layers={args.gnn_layers} hidden={args.gnn_hidden} k={args.knn_k} epochs={args.epochs}", flush=True)

vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0].to(dev) for v in TRAINED}
vte = torch.stack([emb[v].cpu() for v in TRAINED])
cfg = AffordanceGNNConfig(vlm_dim=128, dino_vertex_dim=128, geom_dim=5, sam3d_dim=8, normals_dim=3, pos_dim=0,
    verb_dim=128, num_verbs=8, verb_embedding="text", verb_text_dim=512, verb_in_backbone=True,
    dropout=0.1, gnn_hidden=args.gnn_hidden, gnn_layers=args.gnn_layers, knn_k=args.knn_k)

# Dataset loads features per-__getitem__ (not a global preload); I cache only the fp16 subsample per object,
# so peak memory stays bounded even over 1227 objects. eager=True is needed or vertex_features returns None.
ds = DataRootDataset(manifest_path=args.manifest, load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
obj2row, row2dir = {}, {}
for i, r in enumerate(ds.rows):
    o = os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")); obj2row.setdefault(o, i); row2dir[o] = str(r.sam3d_reconstruction_dir)

def pseudo_path(o, v): return f"{row2dir[o]}/vertex_pseudolabels_{v}.pt"
def has_pseudo(o, v): return os.path.exists(pseudo_path(o, v))
pairs = [(o, v) for o in obj2row for v in TRAINED if has_pseudo(o, v)]
if args.max_objects:
    keep = set(list(obj2row)[:args.max_objects]); pairs = [(o, v) for o, v in pairs if o in keep]
print(f"{len(set(o for o,_ in pairs))} objects, {len(pairs)} (obj,verb) pseudolabel pairs", flush=True)

# lazy subsample + fp16 feature cache (built on first touch, reused across epochs)
cache = {}
def get(o):
    if o not in cache:
        it = ds[obj2row[o]]; V = len(it["vertex_positions"])
        r = np.random.default_rng(abs(hash(o)) % (2**32))
        sel = np.sort(r.choice(V, min(args.sub, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
        knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], args.knn_k)
        feat = {k: (it.get(k).float()[idx].half() if it.get(k) is not None else None)
                for k in ["vertex_features", "dino_vertex_features", "slat_vertex_features", "vertex_normals", "vertex_geom"]}
        cache[o] = (torch.as_tensor(sel, dtype=torch.long), knn, feat)
    return cache[o]
def target(o, v, sel):
    y = torch.as_tensor(np.asarray(torch.load(pseudo_path(o, v), weights_only=False)).reshape(-1), dtype=torch.float32)
    return y[sel].clamp(0, 1)

model = AffordanceGNN(cfg, verb_text_embeddings=vte).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr); pw = torch.tensor(args.pos_weight, device=dev)
print(f"{sum(p.numel() for p in model.parameters()):,} params", flush=True)
t0 = time.time()
for ep in range(args.epochs):
    model.train(); rng.shuffle(pairs); opt.zero_grad(); tot = 0.0
    for i, (o, v) in enumerate(pairs):
        sel, knn, feat = get(o); g = lambda k: (feat[k].float().to(dev) if feat[k] is not None else None)
        logit = model(emb[v], vlm_features=g("vertex_features"), dino_vertex=g("dino_vertex_features"),
                      slat_vertex=g("slat_vertex_features"), vertex_normals=g("vertex_normals"),
                      vertex_geom=g("vertex_geom"), knn_idx=knn.to(dev))
        loss = Fn.binary_cross_entropy_with_logits(logit, target(o, v, sel).to(dev), pos_weight=pw) / 8
        loss.backward(); tot += loss.item() * 8
        if (i + 1) % 8 == 0: opt.step(); opt.zero_grad()
    opt.step(); opt.zero_grad()
    print(f"  ep{ep+1}/{args.epochs} loss={tot/max(1,len(pairs)):.3f} cached={len(cache)} {time.time()-t0:.0f}s", flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.out)
print(f"=== PRETRAIN DONE -> {args.out} ({time.time()-t0:.0f}s) ===", flush=True)
