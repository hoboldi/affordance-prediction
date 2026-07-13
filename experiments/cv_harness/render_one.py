"""Detailed multi-view render of ONE object+verb: GT vs model prediction from several angles, to see
where the model grabs vs where the human labeled. CPU."""
import os, sys, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
OBJ = sys.argv[1] if len(sys.argv) > 1 else "bottle__34_1458_4773__v000"
VERB = sys.argv[2] if len(sys.argv) > 2 else "grasp"
AZIMS = [-60, 20, 110, 200]
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
cg = torch.load("outputs/gnn_k24preclean_fold0.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); e = vlm.encode_text([VERB])[0]
it = ds[o2r[OBJ]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
r = np.random.default_rng(seed_of(OBJ)); sel = np.sort(r.choice(V, min(18000, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
hb = (np.asarray(torch.load(f"human_gt_labels/{OBJ}/vertex_manuallabels_{VERB}.pt", weights_only=False)).reshape(-1)[sel] >= 0.5).astype(int)
sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
with torch.no_grad():
    p = torch.sigmoid(gnn(e, vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
        slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
ap = average_precision_score(hb, p)
binmap = ListedColormap(["#20124d", "#d7263d"])
def draw(ax, c, cmap, vmax, az):
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap=cmap, s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=12, azim=az); ax.set_axis_off()
rows = [("Human GT (grasp)", hb, binmap, 1.0), ("Model prediction", p, "turbo", float(max(0.05, p.max()))),
        ("Prediction @ 0.5 (mask)", (p >= 0.5).astype(int), binmap, 1.0)]
fig = plt.figure(figsize=(len(AZIMS) * 2.4, len(rows) * 2.5))
for ri, (name, c, cmap, vmax) in enumerate(rows):
    for ci, az in enumerate(AZIMS):
        ax = fig.add_subplot(len(rows), len(AZIMS), ri * len(AZIMS) + ci + 1, projection="3d")
        draw(ax, c, cmap, vmax, az)
        if ci == 0: ax.text2D(-0.2, 0.5, name, transform=ax.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center",
                              color=("tab:green" if "Model" in name or "Prediction" in name else "black"))
        if ri == 0: ax.set_title(f"azim {az}°", fontsize=8)
fig.suptitle(f"{OBJ.split('__')[0]} — \"{VERB}\"  (AUPRC={ap:.2f})   rotated views\n"
             "Human labeled the LOWER body as graspable; the model instead predicts the UPPER neck/top — a real GT vs prediction mismatch.", fontsize=10, y=1.01)
out = "outputs/renders/one_bottle_grasp.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out, "AUPRC", round(ap, 3))
