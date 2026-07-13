"""RE-VERIFY on the CLEAN best GNN (gnn_k24preclean_fold0): (1) real verbs vs NONSENSE words vs base-rate,
(2) cross-verb map distinctness. Confirms verb conditioning genuinely routes (not a verb-agnostic prior)
on clean labels + the current model. fold0-val, clean hand-labels. CPU."""
import os, sys, json, hashlib, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
NONSENSE = ["banana", "xqzptv", "the weather", "purple"]
SUB = 20000
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
cg = torch.load("outputs/gnn_k24preclean_fold0.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {w: vlm.encode_text([w])[0] for w in TRAINED + NONSENSE}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
def has(o, v): return os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")
_c = {}
def prep(o):
    if o not in _c:
        it = ds[o2r[o]]; V = len(it["vertex_positions"]); r = np.random.default_rng(seed_of(o))
        sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
        knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k); _c[o] = (it, sel, idx, knn)
    return _c[o]
def pred(o, w):
    it, sel, idx, knn = prep(o); sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        return torch.sigmoid(gnn(emb[w], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
real, nons, base = [], [], []
cross = []
for o in val:
    if o not in o2r: continue
    vs = [v for v in TRAINED if has(o, v)]
    it, sel, idx, knn = None, None, None, None
    maps = {}
    for v in vs:
        raw = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)
        if ((raw > 0.001) & (raw < 0.999)).mean() >= 0.005: continue     # clean only
        _, sel2, _, _ = prep(o); y = (raw[sel2] >= 0.5).astype(int)
        if not (0 < y.sum() < len(y)): continue
        pv = pred(o, v); maps[v] = pv
        real.append(average_precision_score(y, pv)); base.append(y.mean())
        nons.append(np.mean([average_precision_score(y, pred(o, w)) for w in NONSENSE]))
    for i, a in enumerate(list(maps)):                                   # cross-verb distinctness
        for b in list(maps)[i+1:]:
            cross.append(float(np.corrcoef(maps[a], maps[b])[0, 1]))
print(f"=== CLEAN best GNN — verb routing re-verification (fold0-val, n={len(real)} obj-verb) ===")
print(f"  real verb AUPRC     = {np.mean(real):.3f}")
print(f"  NONSENSE word AUPRC = {np.mean(nons):.3f}   (verb-agnostic prior floor)")
print(f"  base-rate (chance)  = {np.mean(base):.3f}")
print(f"  --> verb signal ABOVE nonsense = {np.mean(real)-np.mean(nons):+.3f}; nonsense above chance = {np.mean(nons)-np.mean(base):+.3f}")
print(f"  cross-verb map corr (distinct verbs, lower=more distinct) = {np.mean(cross):+.3f}  (n={len(cross)})")
