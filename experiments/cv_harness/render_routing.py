"""TASK/conditioning demo (clean, no top text): SAME object, different verb -> different region.
GNN prediction (held-out fold) for two objects each under two verbs. Small verb label per cell,
object label at left. Shows verb conditioning genuinely routes the output."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SUB = 24000
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)

ROUTES = [  # (object, label, [(verb, view), (verb, view)])
    ("chair__108_12895_27003__v000", "chair", [("sit", (16, -60)), ("move", (16, -60))]),
    ("cup__20_700_1535__v000",       "cup",   [("pour", (16, -60)), ("grasp", (16, -60))]),
]

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

NC = max(len(vs) for _, _, vs in ROUTES)
fig = plt.figure(figsize=(NC * 2.7, len(ROUTES) * 2.7))
for ri, (o, lab, verbviews) in enumerate(ROUTES):
    k = obj2fold[o]; m, gc = gnn[k]; it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
    knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gc.knn_k)
    sl = lambda kk: (it.get(kk).float()[idx] if it.get(kk) is not None else None)
    for ci, (v, view) in enumerate(verbviews):
        e = vlm.encode_text([v])[0]
        with torch.no_grad():
            p = torch.sigmoid(m(e, vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                    slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
        ax = fig.add_subplot(len(ROUTES), NC, ri * NC + ci + 1, projection="3d")
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=p, cmap="turbo", s=3, vmin=0, vmax=float(max(0.05, p.max())))
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
        ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()
        ax.set_title(f'"{v}"', fontsize=14, fontweight="bold", color="tab:green", pad=-4)
        if ci == 0:
            ax.text2D(-0.06, 0.5, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", rotation=90, va="center")
plt.subplots_adjust(wspace=-0.05, hspace=0.05, left=0.05, right=0.99, top=0.95, bottom=0.02)
out = "outputs/renders/verb_routing_clean.png"; os.makedirs("outputs/renders", exist_ok=True)
fig.savefig(out, dpi=145, bbox_inches="tight"); print("saved", out)
