"""Visual: does geometry sharpen cross-category predictions? Held-out bottles (neither model saw a bottle),
columns = Human GT | no-geom prediction | +geom prediction, rows = (bottle, verb) for contain & grasp. CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

CAT = sys.argv[1] if len(sys.argv) > 1 else "bottle"
VERBS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["contain", "grasp"]
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False); m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval(); return m
nogeom = load(f"outputs/loco_{CAT}/last.pt")
geom = load(f"outputs/locogeom_{CAT}/last.pt")
bcfg = mlp_head_config_from_model_cfg(torch.load(f"outputs/locogeom_{CAT}/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename,
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in VERBS}

def preds(o, v, sel, idx, it):
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    kw = dict(vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
              vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"), slat_vertex=sl("slat_vertex_features"))
    with torch.no_grad():
        png = torch.sigmoid(nogeom(emb[v], **kw)).numpy()
        pg = torch.sigmoid(geom(emb[v], vertex_geom=sl("vertex_geom"), **kw)).numpy()
    return png, pg

# pick 2 objects of CAT that have all the target verbs labeled
sel_objs = [o for o in sorted(o2r) if o.split("__")[0] == CAT
            and all(os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt") for v in VERBS)][:2]
picks = [(o, v) for o in sel_objs for v in VERBS]

fig = plt.figure(figsize=(3 * 2.7, len(picks) * 2.8))
for ri, (o, v) in enumerate(picks):
    it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    sel = rng.choice(V, min(16000, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
    hb = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel]
    png, pg = preds(o, v, sel, idx, it)
    yb = (hb >= 0.5).astype(int)
    a_ng = average_precision_score(yb, png) if 0 < yb.sum() < len(yb) else float("nan")
    a_g = average_precision_score(yb, pg) if 0 < yb.sum() < len(yb) else float("nan")
    cols = [("Human GT", hb, None), (f"no-geom", png, a_ng), (f"+geom", pg, a_g)]
    for ci, (name, fld, ap) in enumerate(cols):
        ax = fig.add_subplot(len(picks), 3, ri * 3 + ci + 1, projection="3d")
        vmax = 1.0 if ci == 0 else float(max(0.05, fld.max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld, cmap="turbo", s=3, vmin=0, vmax=vmax)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
        ax.view_init(elev=16, azim=-60); ax.set_axis_off()
        ttl = name + (f"\nAUPRC={ap:.2f}" if ap is not None else "")
        ax.set_title(ttl, fontsize=9, fontweight=("bold" if ci == 2 else "normal"),
                     color=("tab:green" if ci == 2 else ("gray" if ci == 1 else "black")))
        if ci == 0:
            ax.text2D(-0.15, 0.5, f"{v}\n{CAT}", transform=ax.transAxes, fontsize=11, fontweight="bold", rotation=90, va="center")

fig.suptitle(f"Geometry on cross-category transfer — held-out {CAT}s (neither model saw a {CAT})\n"
             "Human GT | no-geom | +geom.  Geometry helps geometric verbs (contain); can HURT part-based grasp.",
             fontsize=10, y=0.995)
out = f"outputs/renders/geom_transfer_{CAT}.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
for o, v in picks: print(o.split("__")[0], v)
