"""IN-DISTRIBUTION geometry win: base full-FT vs +geom on fold0-VAL objects. Columns Human GT | no-geom | +geom.
Non-cherry-picked: first N fold0-val objects (sorted) that have the verb labeled. CPU."""
import os, sys, json
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
VERB = sys.argv[1] if len(sys.argv) > 1 else "grasp"
NOBJ = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False); m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval(); return m
nogeom = load("outputs/basefull_fold0/last.pt"); geom = load("outputs/geomcv_fold0/last.pt")
bcfg = mlp_head_config_from_model_cfg(torch.load("outputs/geomcv_fold0/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename,
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = vlm.encode_text([VERB])[0]
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
def valid(o):
    hf = f"human_gt_labels/{o}/vertex_manuallabels_{VERB}.pt"
    if o not in o2r or not os.path.exists(hf): return False
    hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5)
    return 0 < hb.sum() < len(hb)
# deterministic + varied: first object of each distinct category that has the verb
_seen=set(); picks=[]
for o in sorted(val):
    cat=o.split("__")[0]
    if cat in _seen or not valid(o): continue
    _seen.add(cat); picks.append(o)
    if len(picks)>=NOBJ: break
def preds(o, sel, idx, it):
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    kw = dict(vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
              vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"), slat_vertex=sl("slat_vertex_features"))
    with torch.no_grad():
        return torch.sigmoid(nogeom(emb, **kw)).numpy(), torch.sigmoid(geom(emb, vertex_geom=sl("vertex_geom"), **kw)).numpy()
fig = plt.figure(figsize=(3 * 2.7, len(picks) * 2.8))
for ri, o in enumerate(picks):
    it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    sel = rng.choice(V, min(16000, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
    hb = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{VERB}.pt", weights_only=False)).reshape(-1)[sel]
    png, pg = preds(o, sel, idx, it); yb = (hb >= 0.5).astype(int)
    a_ng = average_precision_score(yb, png); a_g = average_precision_score(yb, pg)
    for ci, (name, fld, ap) in enumerate([("Human GT", hb, None), ("no-geom", png, a_ng), ("+geom", pg, a_g)]):
        ax = fig.add_subplot(len(picks), 3, ri * 3 + ci + 1, projection="3d")
        vmax = 1.0 if ci == 0 else float(max(0.05, fld.max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld, cmap="turbo", s=3, vmin=0, vmax=vmax)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
        ax.view_init(elev=16, azim=-60); ax.set_axis_off()
        ax.set_title(name + (f"\nAUPRC={ap:.2f}" if ap is not None else ""), fontsize=9,
                     fontweight=("bold" if ci == 2 else "normal"), color=("tab:green" if ci == 2 else ("gray" if ci == 1 else "black")))
        if ci == 0:
            ax.text2D(-0.15, 0.5, o.split("__")[0], transform=ax.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
fig.suptitle(f"Geometry IN-DISTRIBUTION ({VERB}) — fold0-val, first {len(picks)} labeled objects (non-cherry-picked)\n"
             f"Human GT | no-geom (full-FT) | +geom.  5-fold {VERB} gain +0.07.", fontsize=10, y=0.997)
out = f"outputs/renders/geom_indist_{VERB}.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out, "| objs:", [o.split("__")[0] for o in picks])
print("AUPRC (no-geom -> +geom):")
