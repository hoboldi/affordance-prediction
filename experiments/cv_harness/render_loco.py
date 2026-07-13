"""Cross-category generalization: each LOCO model predicts on the category it NEVER trained on.
Rows = (held-out category, verb), cols = Human GT | LOCO prediction. Shows contain/pour transfer,
grasp fails. Clean labels. CPU."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
CATS = [("bottle", ["pour", "grasp"]), ("cup", ["pour"]), ("handbag", ["grasp"])]  # visible regions only (drop interior 'contain')
SUB = 16000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))
allobjs = sorted(o2r)
def has(o, v): return os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")
_c = {}
def prep(o):
    if o not in _c:
        it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
        r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
        _c[o] = (it, xyz, sel, idx)
    return _c[o]
def gt(o, v, sel): return (np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel] >= 0.5).astype(int)

emb = {v: vlm.encode_text([v])[0] for v in set(v for _, vs in CATS for v in vs)}
def predict(m, gcfg, o, v):
    it, xyz, sel, idx = prep(o); knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        return torch.sigmoid(m(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
# for each (cat,verb) pick the REPRESENTATIVE object = closest to the median AUPRC over that category
rows = []; models = {}
for cat, verbs in CATS:
    cg = torch.load(f"outputs/gnn_loco_{cat}_foldx.pt", map_location="cpu", weights_only=False)
    gcfg = AffordanceGNNConfig(**cg["cfg"]); m = AffordanceGNN(gcfg); m.load_state_dict(cg["model"]); m.eval(); models[cat] = (m, gcfg)
    for v in verbs:
        scored = []
        for o in [x for x in allobjs if x.split("__")[0] == cat and has(x, v)]:
            _, _, sel, _ = prep(o); g = gt(o, v, sel)
            if 0 < g.sum() < len(g): scored.append((average_precision_score(g, predict(m, gcfg, o, v)), o))
        if not scored: continue
        scored.sort(); med = scored[len(scored)//2]                       # median-AUPRC object = representative
        rows.append((cat, med[1], v))
def draw(ax, xyz, sel, c, gtcol, title, tc):
    vmax = 1.0 if gtcol else float(max(0.05, c.max()))
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap=("turbo" if not gtcol else "turbo"), s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=16, azim=-60); ax.set_axis_off(); ax.set_title(title, fontsize=9, color=tc, fontweight="bold")
fig = plt.figure(figsize=(2 * 2.7, len(rows) * 2.6))
for ri, (cat, o, v) in enumerate(rows):
    m, gcfg = models[cat]; it, xyz, sel, idx = prep(o)
    knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
    g = gt(o, v, sel); sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        p = torch.sigmoid(m(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
    ap = average_precision_score(g, p) if 0 < g.sum() < len(g) else float("nan")
    tc = "tab:green" if ap >= 0.6 else "tab:red"
    axG = fig.add_subplot(len(rows), 2, ri*2+1, projection="3d"); draw(axG, xyz, sel, g, True, "Human GT" if ri == 0 else "", "black")
    axG.text2D(-0.2, 0.5, f'{cat}\n"{v}"', transform=axG.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
    axP = fig.add_subplot(len(rows), 2, ri*2+2, projection="3d"); draw(axP, xyz, sel, p, False, (f"LOCO pred (unseen category)\nAUPRC={ap:.2f}" if ri == 0 else f"AUPRC={ap:.2f}"), tc)
# (no suptitle — message lives in the poster/paper caption)
out = "outputs/renders/loco_generalization.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out, "rows:", [(c, v) for c, _, v in rows])
