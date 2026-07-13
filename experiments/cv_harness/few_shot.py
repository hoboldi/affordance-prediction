"""Few-shot on a NOVEL verb (press): does a handful of labels let the 6-verb GNN locate the press 'island'?
Base = gnn_k24preclean_fold0 (knows 6 verbs, never saw press). Zero-shot press = base with 'press' text emb
(no press training). Few-shot(K) = clone base, fine-tune on K press objects, eval on held-out press. Curve
over K in {1,3,5,10}, averaged over 3 seeds. Clean human labels. GPU."""
import os, sys, json, glob, argparse, time, collections
sys.path.insert(0, "src")
import numpy as np, torch
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SEEDS = [0, 1, 2]; SUB = 30000; EPOCHS = 20; LR = 3e-4; PW = 5.0
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
ap.add_argument("--verb", default="press"); ap.add_argument("--ks", default="1,3,5,10"); args = ap.parse_args()
NOVEL = args.verb; KS = [int(x) for x in args.ks.split(",")]
dev = torch.device(args.device)
print(f"device={dev} novel={NOVEL} K={KS} seeds={SEEDS}", flush=True)

base = torch.load("outputs/gnn_k24preclean_fold0.pt", map_location="cpu", weights_only=False)
cfg = AffordanceGNNConfig(**base["cfg"])
vlm = VLMWrapper(VLMConfig(device="cpu")); e_press = vlm.encode_text([NOVEL])[0].to(dev)
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
vte = torch.stack([vlm.encode_text([v])[0] for v in TRAINED])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
press_objs = sorted({fp.split("/")[-2] for fp in glob.glob(f"human_gt_labels/*/vertex_manuallabels_{NOVEL}.pt")} & set(o2r))
print(f"{len(press_objs)} {NOVEL} objects with features", flush=True)

cache = {}
def get(o):
    if o not in cache:
        it = ds[o2r[o]]; V = len(it["vertex_positions"])
        r = np.random.default_rng(abs(hash(o)) % (2**32)); sel = np.sort(r.choice(V, min(SUB, V), replace=False))
        idx = torch.as_tensor(sel, dtype=torch.long)
        knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], cfg.knn_k).to(dev)
        feat = {k: (it.get(k).float()[idx].to(dev) if it.get(k) is not None else None)
                for k in ["vertex_features", "dino_vertex_features", "slat_vertex_features", "vertex_normals", "vertex_geom"]}
        cache[o] = (torch.as_tensor(sel, dtype=torch.long), knn, feat)
    return cache[o]
def label(o):
    a = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{NOVEL}.pt", weights_only=False)).reshape(-1)
    return (a >= 0.5).astype(int)
def fwd(model, o):
    sel, knn, ft = get(o)
    return model(e_press, vlm_features=ft["vertex_features"], dino_vertex=ft["dino_vertex_features"],
                 slat_vertex=ft["slat_vertex_features"], vertex_normals=ft["vertex_normals"],
                 vertex_geom=ft["vertex_geom"], knn_idx=knn), sel
def evaluate(model, objs):
    model.eval(); aps = []
    with torch.no_grad():
        for o in objs:
            logit, sel = fwd(model, o); y = label(o)[sel.numpy()]
            if 0 < y.sum() < len(y): aps.append(average_precision_score(y, torch.sigmoid(logit).float().cpu().numpy()))
    return float(np.mean(aps)) if aps else float("nan")
def fresh():
    m = AffordanceGNN(cfg, verb_text_embeddings=vte).to(dev); m.load_state_dict(base["model"]); return m

# zero-shot: base, no press training
zs = evaluate(fresh(), press_objs)
print(f"\nZERO-SHOT {NOVEL} (base, 0 labels) = {zs:.3f}", flush=True)

# few-shot curve
res = collections.defaultdict(list)
for K in KS:
    for s in SEEDS:
        r = np.random.default_rng(1000 + s); perm = list(press_objs); r.shuffle(perm)
        train, test = perm[:K], perm[K:]
        m = fresh(); opt = torch.optim.Adam(m.parameters(), lr=LR); pw = torch.tensor(PW, device=dev)
        for ep in range(EPOCHS):
            m.train(); opt.zero_grad(); tot = 0.0
            for o in train:
                sel, _, _ = get(o); logit, _ = fwd(m, o)
                y = torch.as_tensor(label(o)[sel.numpy()], dtype=torch.float32, device=dev)
                loss = Fn.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
                loss.backward(); tot += loss.item()
            opt.step()
        res[K].append(evaluate(m, test))
    print(f"few-shot K={K:2d}: AUPRC = {np.mean(res[K]):.3f} ± {np.std(res[K]):.3f}  (test n={len(press_objs)-K})", flush=True)
print(f"\n=== FEW-SHOT press curve ===")
print(f"zero-shot(0) = {zs:.3f}")
for K in KS: print(f"  K={K:2d}: {np.mean(res[K]):.3f} (+{np.mean(res[K])-zs:+.3f} vs zero-shot)")
print("=== FEW-SHOT DONE ===", flush=True)
