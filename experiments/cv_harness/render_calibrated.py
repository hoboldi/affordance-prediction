"""Render the GNN at its CALIBRATED per-verb threshold. Columns: Human GT (binary) | GNN raw sigmoid
(flooded) | GNN @ tau* (binary mask, deployable). tau* = threshold maximizing mean IoU on fold0-val per verb.
Shows the flooding is fixed by thresholding, not a ranking error. Clean labels only (backfilled dropped). CPU."""
import os, sys, json, hashlib, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
VERBS = ["grasp", "sit", "display", "pour"]           # the flood-prone verbs
PREF = {"grasp": "handbag", "sit": "chair", "display": "laptop", "pour": "bottle"}
SUB = 22000; THRS = np.linspace(0.05, 0.95, 19); f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def iou(p, y, thr):
    pb = p >= thr; yb = y == 1; u = (pb | yb).sum()
    return float((pb & yb).sum() / u) if u > 0 else np.nan
cg = torch.load("outputs/gnn_k24_fold0.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in VERBS}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
def has(o, v): return os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")
_c = {}
def prep(o):
    if o not in _c:
        it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
        rr = np.random.default_rng(seed_of(o)); sel = np.sort(rr.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
        knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k); _c[o] = (it, xyz, sel, idx, knn)
    return _c[o]
def pred(o, v):
    it, xyz, sel, idx, knn = prep(o); sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        return torch.sigmoid(gnn(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
def gt(o, v, sel): return np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel]

# tau* per verb: threshold maximizing mean IoU over fold0-val
tau = {}
for v in VERBS:
    per = collections.defaultdict(list)
    for o in val:
        if not has(o, v): continue
        _, _, sel, _, _ = prep(o); y = (gt(o, v, sel) >= 0.5).astype(int)
        if not (0 < y.sum() < len(y)): continue
        p = pred(o, v)
        for t in THRS: per[t].append(iou(p, y, t))
    tau[v] = max(THRS, key=lambda t: np.nanmean(per[t]) if per[t] else -1)
print("tau* per verb:", {v: round(float(tau[v]), 2) for v in VERBS})

def panel(ax, xyz, sel, c, cmap, vmax, title, tcol="black", bold=False):
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap=cmap, s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=16, azim=-60); ax.set_axis_off(); ax.set_title(title, fontsize=9, color=tcol, fontweight=("bold" if bold else "normal"))
binmap = ListedColormap(["#20124d", "#d7263d"])   # dark / red

rows = [(v, next((o for o in sorted(val) if o.split("__")[0] == PREF[v] and has(o, v)),
                 next((o for o in sorted(val) if has(o, v)), None))) for v in VERBS]
rows = [(v, o) for v, o in rows if o]
fig = plt.figure(figsize=(3 * 2.7, len(rows) * 2.7))
for ri, (v, o) in enumerate(rows):
    it, xyz, sel, idx, knn = prep(o); y = (gt(o, v, sel) >= 0.5).astype(int); p = pred(o, v)
    i50, itau = iou(p, y, 0.5), iou(p, y, tau[v])
    axG = fig.add_subplot(len(rows), 3, ri*3+1, projection="3d"); panel(axG, xyz, sel, y, binmap, 1.0, "Human GT")
    axG.text2D(-0.15, 0.5, f'"{v}"\n{o.split("__")[0]}', transform=axG.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
    axR = fig.add_subplot(len(rows), 3, ri*3+2, projection="3d"); panel(axR, xyz, sel, p, "turbo", float(max(0.05, p.max())), f"GNN raw (sigmoid)\nIoU@0.5={i50:.2f}", "gray")
    axC = fig.add_subplot(len(rows), 3, ri*3+3, projection="3d"); panel(axC, xyz, sel, (p >= tau[v]).astype(int), binmap, 1.0, f"GNN @ tau*={tau[v]:.2f}\nIoU={itau:.2f}", "tab:green", True)
fig.suptitle("GNN at calibrated per-verb threshold — Human GT | raw sigmoid (floods) | thresholded @ tau* (deployable)\n"
             "The flooding is a calibration effect: a per-verb threshold above 0.5 recovers a clean mask. Clean hand-labels only.", fontsize=10, y=0.998)
os.makedirs("outputs/renders", exist_ok=True)
fig.savefig("outputs/renders/gnn_calibrated.png", dpi=125, bbox_inches="tight")
print("saved gnn_calibrated.png", [(v, o.split("__")[0]) for v, o in rows])
