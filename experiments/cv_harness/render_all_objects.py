"""Comprehensive gallery: current best model (gnn_k24preclean_fold0, GEAL-pretrained clean) on EVERY
held-out fold0-val object, per verb. One figure per verb; each object shown GT | prediction + AUPRC.
Clean hand-labels only. CPU."""
import os, sys, json, hashlib, math, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/gnn_k24preclean_fold0.pt"
OBJ_PER_ROW = 4; SUB = 16000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
cg = torch.load(CKPT, map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
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
def panel(ax, xyz, sel, c, gtcol, title, tcol="black"):
    vmax = 1.0 if gtcol else float(max(0.05, c.max()))
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap="turbo", s=2, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=16, azim=-60); ax.set_axis_off(); ax.set_title(title, fontsize=7, color=tcol)

summary = {}
for v in TRAINED:
    objs = [o for o in sorted(val) if has(o, v) and 0 < (gt(o, v, prep(o)[2]) >= 0.5).sum() < len(prep(o)[2])]
    if not objs: continue
    nrow = math.ceil(len(objs) / OBJ_PER_ROW)
    fig = plt.figure(figsize=(OBJ_PER_ROW * 2 * 1.5, nrow * 1.7))
    aps = []
    for i, o in enumerate(objs):
        it, xyz, sel, idx, knn = prep(o); g = gt(o, v, sel); p = pred(o, v)
        ap = average_precision_score((g >= 0.5).astype(int), p); aps.append(ap)
        r, c = divmod(i, OBJ_PER_ROW); base = r * (OBJ_PER_ROW * 2) + c * 2
        axG = fig.add_subplot(nrow, OBJ_PER_ROW * 2, base + 1, projection="3d"); panel(axG, xyz, sel, g, True, f"{o.split('__')[0]} GT")
        axP = fig.add_subplot(nrow, OBJ_PER_ROW * 2, base + 2, projection="3d"); panel(axP, xyz, sel, p, False, f"pred {ap:.2f}", "tab:green")
    fig.suptitle(f'"{v}" — every held-out object (fold0-val), GEAL-pretrained clean GNN.  n={len(objs)}  mean AUPRC={np.mean(aps):.3f}', fontsize=10, y=0.997)
    out = f"outputs/renders/all_{v}.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    summary[v] = (len(objs), float(np.mean(aps))); print(f"saved {out}  n={len(objs)} meanAUPRC={np.mean(aps):.3f}", flush=True)
print("SUMMARY:", {v: (n, round(a, 3)) for v, (n, a) in summary.items()})
