"""Cherry-pick helper. For every CLEAN (object,verb), predict with the HELD-OUT fold's GNN,
score AUPRC, and render a contact sheet COLORED BY GROUND TRUTH so we can eyeball which examples
have a VISIBLE region (interior verbs like contain-on-bottle look empty -> reject). No titles.
Writes candidates.json (ranked per verb) + contact_<verb>.png. CPU."""
import os, sys, json, hashlib, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
VERBS = ["contain", "sit", "pour", "move", "display", "grasp"]
SUB = 16000
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)

# object -> held-out fold
obj2fold = {}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]:
        obj2fold[o] = k
# 5 held-out GNN models (full-feature GEAL-pretrain = the 0.895 model)
models = {}
for k in range(5):
    cg = torch.load(f"outputs/gnn_full_preclean_fold{k}.pt", map_location="cpu", weights_only=False)
    gcfg = AffordanceGNNConfig(**cg["cfg"]); m = AffordanceGNN(gcfg); m.load_state_dict(cg["model"]); m.eval()
    models[k] = (m, gcfg)

ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in VERBS}

_c = {}
def prep(o):
    if o not in _c:
        it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
        r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False))
        _c[o] = (it, xyz, sel, torch.as_tensor(sel, dtype=torch.long))
    return _c[o]
def rawlab(o, v): return np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)
def predict(o, v):
    m, gcfg = models[obj2fold[o]]; it, xyz, sel, idx = prep(o)
    knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        return torch.sigmoid(m(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()

cand = collections.defaultdict(list)
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in VERBS:
        fp = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(fp): continue
        raw = rawlab(o, v)
        frac_int = float(((raw > 0.001) & (raw < 0.999)).mean())      # contamination proxy
        if frac_int >= 0.005: continue                                 # CLEAN only
        it, xyz, sel, idx = prep(o); y = (raw[sel] >= 0.5).astype(int)
        posfrac = float(y.mean())
        if not (0.01 < posfrac < 0.6): continue                        # avoid empty / whole-object
        ap = float(average_precision_score(y, predict(o, v)))
        cand[v].append({"obj": o, "cat": o.split("__")[0], "ap": round(ap, 3), "posfrac": round(posfrac, 3)})

for v in VERBS: cand[v].sort(key=lambda d: -d["ap"])
json.dump({v: cand[v] for v in VERBS}, open(f"{SCR}/candidates.json", "w"), indent=1)
print("=== ranked candidates per verb (top 6) ===")
for v in VERBS:
    print(f"{v:9s}:", ", ".join(f"{c['cat']}/{c['ap']}" for c in cand[v][:6]))

# ---- contact sheet: rows=verbs, cols=top-8, COLORED BY GT (visible region?) ----
K = 8
fig = plt.figure(figsize=(K * 1.7, len(VERBS) * 1.8))
for ri, v in enumerate(VERBS):
    for ci, c in enumerate(cand[v][:K]):
        o = c["obj"]; it, xyz, sel, idx = prep(o); g = (rawlab(o, v)[sel] >= 0.5).astype(float)
        ax = fig.add_subplot(len(VERBS), K, ri * K + ci + 1, projection="3d")
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=g, cmap="turbo", s=2, vmin=0, vmax=1)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
        ax.view_init(elev=16, azim=-60); ax.set_axis_off()
        ax.set_title(f"{v}:{c['cat']}\n{c['ap']:.2f} #{ci}", fontsize=6)
out = f"{SCR}/contact_sheet.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out)
