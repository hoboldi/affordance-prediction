"""Verb-conditioning showcase for the best model (plain full-FT, out-of-fold). Same object, one column per
verb -> shows the affordance MOVES with the verb (interior vs lip vs handle; seat vs back). Predictions only.
Rows = multi-verb objects (cup/bottle: contain,pour,grasp ; chair: sit,move ; handbag: contain,grasp). CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
rng = np.random.default_rng(1); f = lambda t: t.float() if t is not None else None
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
ALLV = ["contain", "sit", "pour", "move", "display", "grasp"]
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in ALLV}

def verbs_of(o):
    return [v for v in ALLV if os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")]
# pick, per category, an in-coverage object that has the target verb set
WANT = [("cup", ["contain", "pour", "grasp"]), ("bottle", ["contain", "pour", "grasp"]),
        ("chair", ["sit", "move"]), ("handbag", ["contain", "grasp"])]
picks = []
for cat, vs in WANT:
    for o in sorted(o2r):
        if o.split("__")[0] == cat and o in fold_of and all(v in verbs_of(o) for v in vs):
            picks.append((o, vs)); break

def predict(o, v, sel, idx):
    it = ds[o2r[o]]; m = models[fold_of[o]]
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        return torch.sigmoid(m(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                               ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))).numpy()

NC = 3  # max verbs shown per object
fig = plt.figure(figsize=(NC * 2.7, len(picks) * 2.9))
for ri, (o, vs) in enumerate(picks):
    it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    sel = rng.choice(V, min(16000, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
    for ci in range(NC):
        ax = fig.add_subplot(len(picks), NC, ri * NC + ci + 1, projection="3d")
        if ci < len(vs):
            v = vs[ci]; pr = predict(o, v, sel, idx)
            ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=pr, cmap="turbo", s=2, vmin=0, vmax=float(max(0.05, pr.max())))
            ax.set_title(f'verb = "{v}"', fontsize=10, fontweight="bold", color="tab:blue")
        else:
            ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c="lightgray", s=2)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
        ax.view_init(elev=16, azim=-60); ax.set_axis_off()
        if ci == 0:
            ax.text2D(-0.14, 0.5, o.split("__")[0], transform=ax.transAxes, fontsize=12, fontweight="bold", rotation=90, va="center")
    print(f"{o.split('__')[0]:9s} {vs}", flush=True)

fig.suptitle("Verb conditioning — same object, prediction changes with the verb prompt (full-FT, out-of-fold)\n"
             "turbo: blue=0, red=1.  Note the affordance MOVES: interior→lip→handle (cup/bottle), seat→backrest (chair).",
             fontsize=10, y=0.995)
out = "outputs/renders/verb_conditioning.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
