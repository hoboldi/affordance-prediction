"""Render HUMAN GT labels (vertex_manuallabels_<verb>.pt) on meshes for visual verification.
CPU + matplotlib only (no model, no GPU). One row per object, one column per labeled verb,
points colored by the human label value (turbo)."""
import os, glob, collections, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/datasets/customDatasets/cmr2"
SUB = 12000
rng = np.random.default_rng(0)

# gather human labels per object
byobj = collections.defaultdict(dict)
for hf in glob.glob("human_gt_labels/*/vertex_manuallabels_*.pt"):
    o = hf.split("/")[1]; v = hf.split("/")[-1].replace("vertex_manuallabels_", "").replace(".pt", "")
    byobj[o][v] = hf

# pick one representative object per category (the one with the most labeled verbs), preferred verb order
PREF = ["contain", "sit", "grasp", "pour", "press", "display", "move", "lift", "open"]
want_cats = ["bowl", "chair", "bottle", "cup", "vase", "keyboard", "laptop", "microwave"]
chosen = []
for cat in want_cats:
    cands = [(o, vd) for o, vd in byobj.items() if o.split("__")[0] == cat
             and os.path.exists(f"{ROOT}/reconstructions/{o}/vertex_positions.pt")]
    if not cands:
        continue
    o, vd = max(cands, key=lambda ov: len(ov[1]))           # most verbs
    verbs = [v for v in PREF if v in vd][:4]
    if verbs:
        chosen.append((cat, o, verbs))

ncol = max(len(v) for _, _, v in chosen)
fig = plt.figure(figsize=(2.6 * ncol, 2.8 * len(chosen)))
for ri, (cat, o, verbs) in enumerate(chosen):
    xyz = torch.load(f"{ROOT}/reconstructions/{o}/vertex_positions.pt", weights_only=False)
    xyz = np.asarray(xyz)
    sel = rng.choice(len(xyz), min(SUB, len(xyz)), replace=False)
    for ci, v in enumerate(verbs):
        lab = np.asarray(torch.load(byobj[o][v], weights_only=False)).reshape(-1)
        ax = fig.add_subplot(len(chosen), ncol, ri * ncol + ci + 1, projection="3d")
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=lab[sel], cmap="turbo", s=2, vmin=0.0, vmax=1.0)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6
        mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[2] - h, mid[2] + h); ax.set_zlim(mid[1] - h, mid[1] + h)
        ax.view_init(elev=14, azim=-60); ax.set_axis_off()
        posrate = float((lab >= 0.5).mean())
        ax.set_title(f"{v}  ({posrate*100:.0f}% painted)", fontsize=9)
        if ci == 0:
            ax.text2D(-0.15, 0.5, cat, transform=ax.transAxes, fontsize=11, fontweight="bold", rotation=90, va="center")
fig.suptitle("HUMAN GT labels (vertex_manuallabels) — color = human affordance score (turbo: blue=0, red=1)", fontsize=11)
out = "outputs/renders/human_labels_verify.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}  ({len(chosen)} objects: " + ", ".join(f"{c}:{'/'.join(v)}" for c, _, v in chosen) + ")")
