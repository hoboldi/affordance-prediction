"""Representative render of the best CURRENT model = plain full-FT (basefull_fold{0,1,2}, no SLAT).
Score every fold-0/1/2 val object out-of-fold (the fold model that held it out), per verb; pick the
MEDIAN-AUPRC (typical, non-cherry-picked) object per verb; render Human GT | full-FT prediction. CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False); m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval(); return m
models = {k: load(f"outputs/basefull_fold{k}/last.pt") for k in [0, 1, 2]}
fold_of = {}
for k in [0, 1, 2]:
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: fold_of[o] = k
bcfg = mlp_head_config_from_model_cfg(torch.load("outputs/basefull_fold0/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}

def predict(o, v, sub):
    it = ds[o2r[o]]; m = models[fold_of[o]]; V = len(it["vertex_positions"])
    sel = rng.choice(V, min(sub, V), replace=False) if sub else np.arange(V); idx = torch.as_tensor(sel, dtype=torch.long)
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        p = torch.sigmoid(m(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                            ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))).numpy()
    return sel, p

# phase 1: score all fold0-2 val objects, per verb, to find the median object
cand = collections.defaultdict(list)
for o in fold_of:
    if o not in o2r: continue
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        sel, p = predict(o, v, 20000)
        cand[v].append((average_precision_score(hb[sel], p), o))
picks = {}
for v in TRAINED:
    xs = sorted(cand[v]); picks[v] = xs[len(xs) // 2]  # median AUPRC object
    print(f"{v:9s} median AUPRC={picks[v][0]:.3f}  {picks[v][1]}  (from n={len(xs)})", flush=True)

# phase 2: render Human GT | prediction for the median object of each verb
fig = plt.figure(figsize=(2 * 3.0, len(TRAINED) * 2.9))
for ri, v in enumerate(TRAINED):
    ap0, o = picks[v]; it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy()
    hb = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)
    sel, pr = predict(o, v, 16000)
    yb = (hb[sel] >= 0.5).astype(int); ap = average_precision_score(yb, pr) if 0 < yb.sum() < len(yb) else float("nan")
    for ci, (name, fld, show) in enumerate([("Human GT", hb[sel], None), ("full-FT prediction", pr, ap)]):
        ax = fig.add_subplot(len(TRAINED), 2, ri * 2 + ci + 1, projection="3d")
        vmax = 1.0 if ci == 0 else float(max(0.05, fld.max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld, cmap="turbo", s=2, vmin=0, vmax=vmax)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
        ax.view_init(elev=16, azim=-60); ax.set_axis_off()
        ax.set_title(name + (f"\nAUPRC = {show:.2f}" if show is not None else ""), fontsize=9,
                     color=("tab:green" if ci == 1 else "black"), fontweight=("bold" if ci == 1 else "normal"))
        if ci == 0:
            ax.text2D(-0.12, 0.5, f"{v}\n{o.split('__')[0]}", transform=ax.transAxes, fontsize=11, fontweight="bold", rotation=90, va="center")

fig.suptitle("Best current model: full-FT (human-GT fine-tuned, no SLAT) — MEDIAN object per verb, out-of-fold\n"
             "Human GT  |  model prediction     (turbo: blue=0, red=1)     3-fold trained-mean AUPRC ~ 0.81",
             fontsize=10, y=0.995)
out = "outputs/renders/bestmodel_fullft.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
