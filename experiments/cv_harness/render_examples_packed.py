"""Densely-packed gallery of held-out (object, verb) predictions — no empty cells. Diverse categories per
verb (round-robin), current best clean model (GEAL-pretrained GNN). Each panel: prediction + 'verb·category AUPRC'.
Clean hand-labels, fold0-val. CPU."""
import os, sys, json, hashlib, math, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
VERBS = ["contain", "pour", "grasp", "sit", "move", "display"]
CAP = {"contain": 10, "pour": 8, "grasp": 8, "sit": 5, "move": 5, "display": 4}   # ~40 panels
NCOL = 6; SUB = 15000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
cg = torch.load("outputs/gnn_k24preclean_fold0.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in VERBS}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
_c = {}
def prep(o):
    if o not in _c:
        it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
        rr = np.random.default_rng(seed_of(o)); sel = np.sort(rr.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
        knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k); _c[o] = (it, xyz, sel, idx, knn)
    return _c[o]
def valid(o, v):
    if o not in o2r or not os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"): return False
    _, _, sel, _, _ = prep(o); g = (np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel] >= 0.5)
    return 0 < g.sum() < len(g)
def pred(o, v):
    it, xyz, sel, idx, knn = prep(o); sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        return torch.sigmoid(gnn(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
def gt(o, v, sel): return (np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel] >= 0.5).astype(int)

# select diverse examples per verb (round-robin across categories)
selected = []
for v in VERBS:
    objs = [o for o in sorted(val) if valid(o, v)]
    bycat = collections.OrderedDict()
    for o in objs: bycat.setdefault(o.split("__")[0], []).append(o)
    picked = []
    while len(picked) < CAP[v] and any(bycat.values()):
        for cat in list(bycat):
            if bycat[cat]: picked.append(bycat[cat].pop(0))
            if len(picked) >= CAP[v]: break
    selected += [(v, o) for o in picked]
print(f"{len(selected)} example panels")

nrow = math.ceil(len(selected) / NCOL)
fig = plt.figure(figsize=(NCOL * 2.1, nrow * 2.2))
VC = {"contain": "#1f77b4", "pour": "#17becf", "grasp": "#d62728", "sit": "#2ca02c", "move": "#ff7f0e", "display": "#9467bd"}
for i, (v, o) in enumerate(selected):
    it, xyz, sel, idx, knn = prep(o); p = pred(o, v); ap = average_precision_score(gt(o, v, sel), p)
    ax = fig.add_subplot(nrow, NCOL, i + 1, projection="3d")
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=p, cmap="turbo", s=2, vmin=0, vmax=float(max(0.05, p.max())))
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=16, azim=-60); ax.set_axis_off()
    ax.set_title(f'"{v}" · {o.split("__")[0]}\nAUPRC={ap:.2f}', fontsize=8, color=VC[v], fontweight="bold")
fig.suptitle("Held-out affordance predictions across categories & verbs — best model (GEAL-pretrained clean GNN)\n"
             "each panel: predicted affordance for one held-out object + verb (number = AUPRC vs human GT)", fontsize=11, y=1.004)
out = "outputs/renders/examples_packed.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out)
