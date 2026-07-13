"""5-fold CV for AffordanceMLPSlat (SLAT encoder + warm-started DINO trunk), human GT.
Per fold: warm-start from base, freeze the hidden trunk, train SlatEncoder + input layers + verb_proj + out;
eval held-out fold. Reports per-verb + trained-mean vs the 0.650 baseline. Vertex-subsampled for speed."""
import os, sys, json, argparse, time, dataclasses
sys.path.insert(0, "src")
import numpy as np, torch
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import mlp_head_config_from_model_cfg
from models.slat_encoder import SlatEncoderConfig, build_slat_knn, build_vertex_to_slat, build_vertex_to_slat_knn
from models.mlp_slat import AffordanceMLPSlat, load_base_into_slat_model
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
BASE_CV = {"contain": 0.748, "sit": 0.818, "pour": 0.762, "move": 0.707, "display": 0.543, "grasp": 0.321}
BASE = "outputs/ov_concat_finepatch/best.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--sub", type=int, default=30000)
ap.add_argument("--pos_weight", type=float, default=5.0)
ap.add_argument("--slat_out", type=int, default=64)
ap.add_argument("--max_objects", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--tag", default="slat")
ap.add_argument("--full_ft", action="store_true")          # unfreeze the warm-started trunk
ap.add_argument("--drop_clip", action="store_true")        # zero the (dead) CLIP-vision channel
ap.add_argument("--enc_hidden", type=int, default=64)      # SlatEncoder width
ap.add_argument("--enc_layers", type=int, default=3)       # SlatEncoder depth
ap.add_argument("--vertex_agg", default="nearest")         # voxel->vertex: nearest|mean|max|wsum
ap.add_argument("--vertex_kv", type=int, default=8)        # #latents aggregated per vertex (if not nearest)
args = ap.parse_args()
dev = torch.device(args.device); rng = np.random.default_rng(0)

base_ck = torch.load(BASE, map_location="cpu", weights_only=False)
base_cfg = mlp_head_config_from_model_cfg(base_ck["model_cfg"]); vte = base_ck["model"].get("verb_text")
slat_cfg = SlatEncoderConfig(in_dim=8, hidden=args.enc_hidden, layers=args.enc_layers, out_dim=args.slat_out, knn_k=8,
                             vertex_agg=args.vertex_agg, vertex_kv=args.vertex_kv)
mlp_cfg = dataclasses.replace(base_cfg, sam3d_dim=args.slat_out)

ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=base_cfg.dino_filename)
obj2row = {}
for i, r in enumerate(ds.rows):
    obj2row.setdefault(os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")), i)
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0].to(dev) for v in TRAINED}

slat_cache = {}
def get_slat(o):
    if o not in slat_cache:
        rd = f"{ROOT}/reconstructions/{o}"
        feats = torch.load(f"{rd}/slat_feats.pt", weights_only=False).float()
        coords = torch.load(f"{rd}/slat_coords.pt", weights_only=False).float()
        pos = torch.as_tensor(np.asarray(torch.load(f"{rd}/vertex_positions.pt", weights_only=False)), dtype=torch.float32)
        if args.vertex_agg == "nearest":
            v2s = build_vertex_to_slat(pos, coords).to(dev); vw = None
        else:
            _i, _w = build_vertex_to_slat_knn(pos, coords, args.vertex_kv); v2s = _i.to(dev); vw = _w.to(dev)
        slat_cache[o] = (feats.to(dev), coords.to(dev), build_slat_knn(coords, 8).to(dev), v2s, vw)
    return slat_cache[o]

def verbs_of(o):
    return [v for v in TRAINED if os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")]
def human(o, v):
    return torch.as_tensor(np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1), dtype=torch.float32)

def forward(model, o, v, sel):
    it = ds[obj2row[o]]
    idx = torch.as_tensor(sel, dtype=torch.long)
    sl = lambda k: (it.get(k).float()[idx].to(dev) if it.get(k) is not None else None)   # per-vertex: subsample on CPU then move
    g = lambda k: (it.get(k).float().to(dev) if it.get(k) is not None else None)          # global tokens
    sf, sc, sk, v2s, vw = get_slat(o)
    vlm = sl("vertex_features")
    if args.drop_clip and vlm is not None: vlm = torch.zeros_like(vlm)   # ablation: kill dead CLIP-vision
    isub = idx.to(dev)
    return model(emb[v], slat_feats=sf, slat_coords=sc, slat_knn=sk, vertex_to_slat=v2s[isub],
                 vertex_weights=(vw[isub] if vw is not None else None),
                 vlm_features=vlm, dino_vertex=sl("dino_vertex_features"),
                 dino_cls=g("dino_cls"), ss_dino_cls=g("ss_dino_cls"), vertex_normals=sl("vertex_normals"))

results = []   # (verb, obj, ap) per held-out (object, verb)
for k in [int(x) for x in args.folds.split(",")]:
    split = json.load(open(f"{SCR}/cv_fold{k}.json"))
    train_objs = [o for o in split["train"] if o in obj2row]
    val_objs = [o for o in split["val"] if o in obj2row]
    if args.max_objects:
        train_objs = train_objs[:args.max_objects]; val_objs = val_objs[:max(5, args.max_objects // 3)]
    model = AffordanceMLPSlat(mlp_cfg, slat_cfg, verb_text_embeddings=vte).to(dev)
    load_base_into_slat_model(model, BASE)
    trn = []
    for n, p in model.named_parameters():
        p.requires_grad_(True if args.full_ft else (("slat_encoder" in n) or any(t in n for t in ["in_norm", "in_proj", "verb_proj", ".out"])))
        if p.requires_grad: trn.append(p)
    opt = torch.optim.Adam(trn, lr=args.lr); pw = torch.tensor(args.pos_weight, device=dev)
    pairs = [(o, v) for o in train_objs for v in verbs_of(o)]
    print(f"fold {k}: {len(train_objs)} train obj, {len(pairs)} pairs, {sum(p.numel() for p in trn):,} trainable", flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(pairs); opt.zero_grad(); tot = 0.0
        for i, (o, v) in enumerate(pairs):
            hb = human(o, v); V = len(hb); sel = rng.choice(V, min(args.sub, V), replace=False)
            logit = forward(model, o, v, sel)
            loss = Fn.binary_cross_entropy_with_logits(logit, hb[sel].to(dev).clamp(0, 1), pos_weight=pw) / 8
            loss.backward(); tot += loss.item() * 8
            if (i + 1) % 8 == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        print(f"  fold{k} ep{ep+1}/{args.epochs} loss={tot/max(1,len(pairs)):.3f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        for o in val_objs:
            for v in verbs_of(o):
                hb = human(o, v); V = len(hb); sel = rng.choice(V, min(25000, V), replace=False)
                yb = (hb[sel].numpy() >= 0.5).astype(int)
                if yb.sum() == 0 or (yb == 1).all(): continue
                p = torch.sigmoid(forward(model, o, v, sel)).float().cpu().numpy()
                results.append((v, o, float(average_precision_score(yb, p))))
    torch.save({"model": model.state_dict()}, f"outputs/cv_{args.tag}_fold{k}.pt")
    print(f"=== fold {k} done ({time.time()-t0:.0f}s) ===", flush=True)

import collections
json.dump(results, open(f"/tmp/slat_cv_{args.tag}.json", "w"))   # per-(verb,obj,ap) for bootstrap
byv = collections.defaultdict(list)
for v, o, a in results:
    byv[v].append(a)
print(f"\n=== SLAT-encoder CV (out-of-fold, human GT) vs baseline ===")
means = []
for v in TRAINED:
    if byv[v]:
        m = float(np.mean(byv[v])); means.append(m)
        print(f"  {v:9s} n={len(byv[v]):3d}  SLAT={m:.3f}  base={BASE_CV[v]:.3f}  d={m-BASE_CV[v]:+.3f}")
if means:
    print(f"  TRAINED-MEAN SLAT={np.mean(means):.3f}  base=0.650  d={np.mean(means)-0.650:+.3f}")
print("=== SLAT CV DONE ===", flush=True)
