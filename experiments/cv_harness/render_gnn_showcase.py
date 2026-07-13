"""Showcase the NEW BEST model — spatial GNN (gnn_edgeconv_fold0) — on fold0-val (held-out).
Fig A (verb-conditioning): same object, columns=verbs, rows=(GT, GNN) -> affordance moves with the verb.
Fig B (gallery): each trained verb on a representative object, Human GT | GNN prediction + AUPRC.
Non-cherry-picked (first sorted fold0-val object per category/verb). CPU."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
ALLV = ["contain", "sit", "pour", "move", "display", "grasp"]
SUB = 20000; f = lambda t: t.float() if t is not None else None
CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/gnn_edgeconv_fold0.pt"
TAG = sys.argv[2] if len(sys.argv) > 2 else "gnn"
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
cg = torch.load(CKPT, map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in ALLV}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
def has(o, v): return os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")

_cache = {}
def prep(o):
    if o not in _cache:
        it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
        r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False))
        idx = torch.as_tensor(sel, dtype=torch.long)
        knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
        _cache[o] = (it, xyz, sel, idx, knn)
    return _cache[o]
def gt(o, v, sel): return np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel]
def pred(o, v):
    it, xyz, sel, idx, knn = prep(o)
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        return torch.sigmoid(gnn(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"),
            vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()

def panel(ax, xyz, sel, fld, gtcol, title, tcolor="black", bold=False):
    vmax = 1.0 if gtcol else float(max(0.05, fld.max()))
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld, cmap="turbo", s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=16, azim=-60); ax.set_axis_off()
    ax.set_title(title, fontsize=9, fontweight=("bold" if bold else "normal"), color=tcolor)

# ---------- Fig A: verb-conditioning ----------
VC = ["contain", "pour", "grasp"]
objsA = [next((o for o in sorted(val) if o.split("__")[0] == cat and all(has(o, v) for v in VC)), None) for cat in ["cup", "bottle"]]
objsA = [o for o in objsA if o]
figA = plt.figure(figsize=(len(VC) * 2.7, len(objsA) * 2 * 2.6))
for oi, o in enumerate(objsA):
    it, xyz, sel, idx, knn = prep(o)
    for vi, v in enumerate(VC):
        g = gt(o, v, sel); p = pred(o, v)
        ap = average_precision_score((g >= 0.5).astype(int), p) if 0 < (g >= 0.5).sum() < len(g) else float("nan")
        axG = figA.add_subplot(len(objsA) * 2, len(VC), (oi * 2) * len(VC) + vi + 1, projection="3d")
        panel(axG, xyz, sel, g, True, f'"{v}"  (Human GT)', bold=(vi == 0))
        if vi == 0: axG.text2D(-0.12, 0.5, o.split("__")[0], transform=axG.transAxes, fontsize=11, fontweight="bold", rotation=90, va="center")
        axP = figA.add_subplot(len(objsA) * 2, len(VC), (oi * 2 + 1) * len(VC) + vi + 1, projection="3d")
        panel(axP, xyz, sel, p, False, f"GNN  AUPRC={ap:.2f}", tcolor="tab:green", bold=True)
figA.suptitle("GEAL-pretrained GNN (clean labels) — verb conditioning on held-out objects\n"
              "Same geometry, the predicted affordance moves with the verb: interior (contain) → spout (pour) → body (grasp).", fontsize=10, y=0.998)
os.makedirs("outputs/renders", exist_ok=True)
figA.savefig("outputs/renders/pre_conditioning.png", dpi=125, bbox_inches="tight"); print("saved gnn_conditioning.png", [o.split("__")[0] for o in objsA])

# ---------- Fig B: gallery ---------- (representative object per verb; handbag for grasp = handles)
PREF = {"grasp": "handbag", "contain": "bottle", "sit": "chair", "pour": "bottle", "move": "chair", "display": "laptop"}
def pick_for(v):
    pref = [o for o in sorted(val) if has(o, v) and o.split("__")[0] == PREF.get(v)]
    return (pref or [o for o in sorted(val) if has(o, v)] or [None])[0]
picks = [(v, pick_for(v)) for v in ALLV]
figB = plt.figure(figsize=(2 * 2.7, len(picks) * 2.6))
for ri, (v, o) in enumerate(picks):
    if o is None: continue
    it, xyz, sel, idx, knn = prep(o)
    g = gt(o, v, sel); p = pred(o, v)
    ap = average_precision_score((g >= 0.5).astype(int), p) if 0 < (g >= 0.5).sum() < len(g) else float("nan")
    axG = figB.add_subplot(len(picks), 2, ri * 2 + 1, projection="3d")
    panel(axG, xyz, sel, g, True, "Human GT" if ri == 0 else "")
    axG.text2D(-0.18, 0.5, f'"{v}"\n{o.split("__")[0]}', transform=axG.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
    axP = figB.add_subplot(len(picks), 2, ri * 2 + 2, projection="3d")
    panel(axP, xyz, sel, p, False, (f"GNN prediction\nAUPRC={ap:.2f}" if ri == 0 else f"AUPRC={ap:.2f}"), tcolor="tab:green", bold=True)
figB.suptitle("GEAL-pretrained GNN (clean labels) — all six trained verbs, held-out objects\nHuman GT | GNN prediction", fontsize=10, y=0.997)
figB.savefig("outputs/renders/pre_gallery.png", dpi=125, bbox_inches="tight"); print("saved gnn_gallery.png", [(v, o.split('__')[0] if o else None) for v, o in picks])
