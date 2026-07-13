"""Full pipeline example rows: [reference image | reconstructed mesh | GEAL pseudolabel | human label |
lean-GNN prediction] for candidate (obj,verb) so we can pick the best to replace the bowl demo.
Held-out fold model for the prediction. CPU."""
import sys, os, json
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
DR = "/home/datasets/customDatasets/cmr2"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SUB = 24000

CANDS = [  # (obj, verb, view)
    ("cup__20_685_1352__v000",       "grasp",   (16, -60)),
    ("bottle__34_1420_4409__v000",   "pour",    (16, -60)),
    ("laptop__122_14291_29271__v000","display", (25, 110)),
    ("chair__108_12895_27003__v000", "sit",     (16, -60)),
    ("vase__108_12867_22800__v000",  "pour",    (16, -60)),
]

def lf(p):
    x = torch.load(p, map_location="cpu", weights_only=False)
    return (x["features"] if isinstance(x, dict) and "features" in x else x)
obj2fold = {}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: obj2fold[o] = k
gnn = {}
for k in range(5):
    cg = torch.load(f"outputs/gnn_lean_preclean_fold{k}.pt", map_location="cpu", weights_only=False)
    c = AffordanceGNNConfig(**cg["cfg"]); mm = AffordanceGNN(c); mm.load_state_dict(cg["model"]); mm.eval(); gnn[k] = (mm, c)
vlm = VLMWrapper(VLMConfig(device="cpu"))

def predict(o, v, sel, idx):
    k = obj2fold.get(o, 0); m, c = gnn[k]; R = f"{DR}/reconstructions/{o}"
    dino = lf(f"{R}/vertex_dino_fine.pt").float()[idx]; geom = lf(f"{R}/vertex_geom.pt").float()[idx]
    pos = lf(f"{R}/vertex_positions.pt").float()[idx]
    knn = AffordanceGNN._build_knn(pos, c.knn_k)
    z = lambda d: torch.zeros(len(sel), d)
    with torch.no_grad():
        return torch.sigmoid(m(vlm.encode_text([v])[0], vlm_features=z(c.vlm_dim), dino_vertex=dino,
            slat_vertex=z(c.sam3d_dim), vertex_normals=z(c.normals_dim), vertex_geom=geom, knn_idx=knn)).numpy()

def draw3d(ax, xyz, sel, c, view, binary=False):
    vmax = 1.0 if binary else float(max(0.05, c.max()))
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap="turbo", s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0)-xyz[sel].min(0)).max()/2+1e-6; mid = (xyz[sel].max(0)+xyz[sel].min(0))/2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()

COLS = ["input image", "reconstruction", "GEAL (teacher)", "human GT", "prediction (ours)"]
fig = plt.figure(figsize=(5 * 2.5, len(CANDS) * 2.6))
for ri, (o, v, view) in enumerate(CANDS):
    R = f"{DR}/reconstructions/{o}"
    xyz = lf(f"{R}/vertex_positions.pt").float().numpy(); V = len(xyz)
    r = np.random.default_rng(0); sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
    geal = np.asarray(lf(f"{R}/vertex_pseudolabels_{v}.pt")).reshape(-1)[sel]
    hum = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel]
    pred = predict(o, v, sel, idx)
    # panel 1: image
    ax = fig.add_subplot(len(CANDS), 5, ri*5+1); img = Image.open(f"{DR}/sam3d_inputs/images/{o}.png").convert("RGB")
    ax.imshow(img); ax.set_axis_off()
    ax.text(-0.12, 0.5, f'{o.split("__")[0]}\n"{v}"', transform=ax.transAxes, fontsize=11, fontweight="bold", rotation=90, va="center")
    # panels 2-5
    for ci, (fld, binary) in enumerate([(xyz[sel,1], False), (geal, False), (hum >= 0.5, True), (pred, False)]):
        a = fig.add_subplot(len(CANDS), 5, ri*5+2+ci, projection="3d")
        draw3d(a, xyz, sel, (fld.astype(float) if binary else fld), view, binary=binary)
        if ci == 0:  # reconstruction = solid light-gray geometry
            a.collections[0].set_array(None); a.collections[0].set_color("#8a8f98")
    if ri == 0:
        for ci in range(5):
            fig.axes[ri*5+ci].set_title(COLS[ci], fontsize=11, fontweight="bold", color=("tab:green" if ci==4 else "black"))
out = f"{SCR}/pipeline_candidates.png"; fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
