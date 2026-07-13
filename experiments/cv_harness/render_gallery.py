"""Gallery of good GT<->prediction pairs across diverse verb-object combos. Each cell pair = Human GT | GNN,
held-out fold model, clean labels, visible regions. 3 pairs/row. Small verb:cat label + AUPRC; no suptitle."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SUB = 20000
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)

GALLERY = [  # (verb, object, view) — 2x4, all 6 trained verbs, diverse categories (grasp/pour on 2 diff cats)
    ("contain", "bowl__66_4878_11834__v000",     (16, -60)),   # rim
    ("sit",     "chair__108_12895_27003__v000",  (16, -60)),   # seat
    ("pour",    "bottle__34_1420_4409__v000",    (16, -60)),   # spout
    ("grasp",   "handbag__20_698_1364__v000",    (16, -60)),   # handle
    ("display", "laptop__122_14291_29271__v000", (25, 110)),   # screen
    ("move",    "chair__108_12868_22877__v000",  (16, -60)),   # back
    ("grasp",   "cup__20_685_1352__v000",        (16, -60)),   # mug handle
    ("pour",    "vase__108_12867_22800__v000",   (16, -60)),   # rim
]
PAIRS_PER_ROW = 4

obj2fold = {}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: obj2fold[o] = k
gnn = {}
for k in range(5):
    cg = torch.load(f"outputs/gnn_full_preclean_fold{k}.pt", map_location="cpu", weights_only=False)
    gc = AffordanceGNNConfig(**cg["cfg"]); m = AffordanceGNN(gc); m.load_state_dict(cg["model"]); m.eval(); gnn[k] = (m, gc)
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))

def draw(ax, xyz, sel, c, gt, view):
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap="turbo", s=3, vmin=0, vmax=(1.0 if gt else float(max(0.05, c.max()))))
    h = (xyz[sel].max(0)-xyz[sel].min(0)).max()/2+1e-6; mid = (xyz[sel].max(0)+xyz[sel].min(0))/2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()

nrows = (len(GALLERY) + PAIRS_PER_ROW - 1) // PAIRS_PER_ROW
ncols = PAIRS_PER_ROW * 2
fig = plt.figure(figsize=(ncols * 1.75, nrows * 2.3))
for i, (v, o, view) in enumerate(GALLERY):
    k = obj2fold[o]; m, gc = gnn[k]; it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
    knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gc.knn_k); sl = lambda kk: (it.get(kk).float()[idx] if it.get(kk) is not None else None)
    gt = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel]
    with torch.no_grad():
        p = torch.sigmoid(m(vlm.encode_text([v])[0], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
    ap = average_precision_score((gt >= 0.5).astype(int), p)
    rr, cc = divmod(i, PAIRS_PER_ROW); base = rr * ncols + cc * 2
    axg = fig.add_subplot(nrows, ncols, base + 1, projection="3d"); draw(axg, xyz, sel, gt, True, view)
    axg.set_title(f"{v} · {o.split('__')[0]}", fontsize=9, fontweight="bold", pad=-1)
    axp = fig.add_subplot(nrows, ncols, base + 2, projection="3d"); draw(axp, xyz, sel, p, False, view)
    axp.set_title(f"AUPRC {ap:.2f}", fontsize=8, color="tab:green", pad=-1)
plt.subplots_adjust(wspace=-0.05, hspace=0.12, left=0.01, right=0.99, top=0.96, bottom=0.01)
out = "outputs/renders/gallery_clean.png"; os.makedirs("outputs/renders", exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out, "| GT|GNN pairs:", len(GALLERY))
