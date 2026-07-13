"""One representative image: rows = one object per category, columns = verbs. Each cell shows the current
best model's predicted affordance (turbo) where the verb applies to that category (+AUPRC), or the object
in flat gray where it doesn't. Held-out fold0-val, clean labels, GEAL-pretrained best model. CPU."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
VERBS = ["contain", "pour", "grasp", "sit", "move", "display"]   # logical order
SUB = 17000; f = lambda t: t.float() if t is not None else None
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
    if not os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"): return False
    _, _, sel, _, _ = prep(o); g = (np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel] >= 0.5)
    return 0 < g.sum() < len(g)
def pred(o, v):
    it, xyz, sel, idx, knn = prep(o); sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        return torch.sigmoid(gnn(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
def gt(o, v, sel): return (np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel] >= 0.5).astype(int)

# one representative object per category: the one with the most relevant verbs (first-sorted tiebreak)
cats = sorted(set(o.split("__")[0] for o in val if o in o2r))
rep = {}
for cat in cats:
    best, bestn = None, 0
    for o in sorted(val):
        if o.split("__")[0] != cat or o not in o2r: continue
        n = sum(valid(o, v) for v in VERBS)
        if n > bestn: bestn, best = n, o
    if best: rep[cat] = best
rows = [(cat, rep[cat]) for cat in cats if cat in rep]
print("rows:", [(c, o.split("__")[0]) for c, o in rows])

def draw(ax, xyz, sel, c, cmap, vmax, title, tcol):
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap=cmap, s=2, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=16, azim=-60); ax.set_axis_off()
    if title: ax.set_title(title, fontsize=8, color=tcol, fontweight="bold")

nrow, ncol = len(rows), len(VERBS)
fig = plt.figure(figsize=(ncol * 1.9, nrow * 2.0))
for ri, (cat, o) in enumerate(rows):
    it, xyz, sel, idx, knn = prep(o)
    for ci, v in enumerate(VERBS):
        ax = fig.add_subplot(nrow, ncol, ri * ncol + ci + 1, projection="3d")
        if valid(o, v):
            p = pred(o, v); ap = average_precision_score(gt(o, v, sel), p)
            draw(ax, xyz, sel, p, "turbo", float(max(0.05, p.max())), f"{ap:.2f}", "tab:green")
        else:
            draw(ax, xyz, sel, np.ones(len(sel)), "Greys", 1.6, "", "gray")   # flat gray = verb N/A
        if ci == 0: ax.text2D(-0.25, 0.5, cat, transform=ax.transAxes, fontsize=11, fontweight="bold", rotation=90, va="center")
        if ri == 0: ax.text2D(0.5, 1.15, f'"{v}"', transform=ax.transAxes, fontsize=12, fontweight="bold", ha="center", color="black")
fig.suptitle("Affordance across categories × verbs — one representative held-out object per category (best model: GEAL-pretrained clean GNN)\n"
             "colored = predicted affordance (number = AUPRC vs human GT) · gray = verb not applicable to that object", fontsize=11, y=1.005)
out = "outputs/renders/category_verb_matrix.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
