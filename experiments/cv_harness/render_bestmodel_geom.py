"""Showcase the CURRENT BEST model (full-FT + geometry, geomcv_fold0) on fold0-val (held-out) objects.
Fig A (verb-conditioning): same object, columns = verbs -> affordance MOVES with the verb (Human GT vs pred).
Fig B (gallery): each trained verb on a representative object, Human GT | +geom prediction + AUPRC.
Non-cherry-picked: first sorted fold0-val object per category/verb. CPU (GPU is busy)."""
import os, sys, json
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
ALLV = ["contain", "sit", "pour", "move", "display", "grasp"]
rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
c = torch.load("outputs/geomcv_fold0/last.pt", map_location="cpu", weights_only=False)
m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval()
bcfg = mlp_head_config_from_model_cfg(c["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename,
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in ALLV}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]

def has(o, v): return os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")
def gt(o, v, sel): return np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel]
def pred(o, v, sel, idx, it):
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    kw = dict(vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
              vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"),
              slat_vertex=sl("slat_vertex_features"), vertex_geom=sl("vertex_geom"))
    with torch.no_grad():
        return torch.sigmoid(m(emb[v], **kw)).numpy()

def panel(ax, xyz, sel, fld, gtcol, title, tcolor="black", bold=False):
    vmax = 1.0 if gtcol else float(max(0.05, fld.max()))
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld, cmap="turbo", s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=16, azim=-60); ax.set_axis_off()
    ax.set_title(title, fontsize=9, fontweight=("bold" if bold else "normal"), color=tcolor)

# ---------- Fig A: verb-conditioning (same object, verb moves the map) ----------
VC = ["contain", "pour", "grasp"]
objsA = []
for cat in ["cup", "bottle"]:
    cand = [o for o in sorted(val) if o.split("__")[0] == cat and all(has(o, v) for v in VC)]
    if cand: objsA.append(cand[0])
figA = plt.figure(figsize=(len(VC) * 2.7, len(objsA) * 2 * 2.6))
for oi, o in enumerate(objsA):
    it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    sel = rng.choice(V, min(16000, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
    for vi, v in enumerate(VC):
        g = gt(o, v, sel); p = pred(o, v, sel, idx, it)
        ap = average_precision_score((g >= 0.5).astype(int), p) if 0 < (g >= 0.5).sum() < len(g) else float("nan")
        rG = (oi * 2) * len(VC) + vi + 1; rP = (oi * 2 + 1) * len(VC) + vi + 1
        axG = figA.add_subplot(len(objsA) * 2, len(VC), rG, projection="3d")
        panel(axG, xyz, sel, g, True, f'"{v}"  (Human GT)', tcolor="black", bold=(vi == 0))
        if vi == 0: axG.text2D(-0.12, 0.5, o.split("__")[0], transform=axG.transAxes, fontsize=11, fontweight="bold", rotation=90, va="center")
        axP = figA.add_subplot(len(objsA) * 2, len(VC), rP, projection="3d")
        panel(axP, xyz, sel, p, False, f"prediction  AUPRC={ap:.2f}", tcolor="tab:green", bold=True)
figA.suptitle("Best model (full-FT + geometry) — verb conditioning on held-out objects\n"
              "Same geometry, the predicted affordance MOVES with the verb: interior (contain) → lip/spout (pour) → body/handle (grasp).",
              fontsize=10, y=0.998)
os.makedirs("outputs/renders", exist_ok=True)
figA.savefig("outputs/renders/best_conditioning.png", dpi=125, bbox_inches="tight"); print("saved best_conditioning.png", [o.split("__")[0] for o in objsA])

# ---------- Fig B: gallery across all 6 trained verbs ----------
picks = []
for v in ALLV:
    cand = [o for o in sorted(val) if has(o, v)]
    picks.append((v, cand[0]) if cand else (v, None))
figB = plt.figure(figsize=(2 * 2.7, len(picks) * 2.6))
for ri, (v, o) in enumerate(picks):
    if o is None: continue
    it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    sel = rng.choice(V, min(16000, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
    g = gt(o, v, sel); p = pred(o, v, sel, idx, it)
    ap = average_precision_score((g >= 0.5).astype(int), p) if 0 < (g >= 0.5).sum() < len(g) else float("nan")
    axG = figB.add_subplot(len(picks), 2, ri * 2 + 1, projection="3d")
    panel(axG, xyz, sel, g, True, "Human GT" if ri == 0 else "", bold=False)
    axG.text2D(-0.18, 0.5, f'"{v}"\n{o.split("__")[0]}', transform=axG.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
    axP = figB.add_subplot(len(picks), 2, ri * 2 + 2, projection="3d")
    panel(axP, xyz, sel, p, False, (f"+geom prediction\nAUPRC={ap:.2f}" if ri == 0 else f"AUPRC={ap:.2f}"), tcolor="tab:green", bold=True)
figB.suptitle("Best model (full-FT + geometry) — all six trained verbs, held-out objects\nHuman GT | prediction", fontsize=10, y=0.997)
figB.savefig("outputs/renders/best_gallery.png", dpi=125, bbox_inches="tight"); print("saved best_gallery.png", [(v, o.split('__')[0] if o else None) for v, o in picks])
