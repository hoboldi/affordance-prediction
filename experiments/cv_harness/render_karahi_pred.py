"""Run the LEAN final GNN on the karahi demo mesh and render fresh affordance predictions.
Lean model = DINO + geometry + verb-text; the dropped channels (clip/slat/normals) are fed zeros,
exactly as lean was trained. Produces a single 'contain' panel (to replace 04_prediction.png) and a
2-panel contain+grasp version. CPU."""
import sys, os
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.mesh_loading import load_mesh
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

D = "cam_karahi_feat/karahi__demo__v000"
SUB = 30000
def loadfeat(p):
    x = torch.load(p, map_location="cpu", weights_only=False)
    return (x["features"] if isinstance(x, dict) and "features" in x else x).float()

cg = torch.load("outputs/gnn_lean_preclean_fold0.pt", map_location="cpu", weights_only=False)
cfg = AffordanceGNNConfig(**cg["cfg"]); m = AffordanceGNN(cfg); m.load_state_dict(cg["model"]); m.eval()

mesh = load_mesh(f"{D}/mesh.glb", process=False)
xyz = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32); V = len(xyz)
dino = loadfeat(f"{D}/vertex_dino_fine.pt"); geom = loadfeat(f"{D}/vertex_geom.pt")
assert dino.shape[0] == V and geom.shape[0] == V, f"feat/vertex mismatch dino={dino.shape} geom={geom.shape} V={V}"

r = np.random.default_rng(0); sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
pos = xyz[idx].numpy()
knn = AffordanceGNN._build_knn(xyz[idx], cfg.knn_k)
dino_s, geom_s = dino[idx], geom[idx]
zc = torch.zeros(len(sel), cfg.vlm_dim); zs = torch.zeros(len(sel), cfg.sam3d_dim); zn = torch.zeros(len(sel), cfg.normals_dim)
vlm = VLMWrapper(VLMConfig(device="cpu"))

def predict(verb):
    e = vlm.encode_text([verb])[0]
    with torch.no_grad():
        p = torch.sigmoid(m(e, vlm_features=zc, dino_vertex=dino_s, slat_vertex=zs,
                            vertex_normals=zn, vertex_geom=geom_s, knn_idx=knn)).numpy()
    print(f"  {verb}: pred range [{p.min():.2f},{p.max():.2f}] frac>0.5={ (p>0.5).mean():.2f}")
    return p

VIEW = (42, -70)
def draw(ax, c, title):
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=c, cmap="turbo", s=4, vmin=0, vmax=float(max(0.05, c.max())))
    h = (pos.max(0) - pos.min(0)).max() / 2 + 1e-6; mid = (pos.max(0) + pos.min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=VIEW[0], azim=VIEW[1]); ax.set_axis_off()
    if title: ax.set_title(title, fontsize=13, fontweight="bold", color="tab:green")

pc = predict("contain"); pg = predict("grasp")

# (1) single 'contain' panel — replaces 04_prediction.png (matches old single-object style)
fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="3d"); draw(ax, pc, "")
fig.savefig("cam_karahi_feat/karahi_pred_contain.png", dpi=150, bbox_inches="tight", transparent=True)
# (2) contain + grasp
fig2 = plt.figure(figsize=(9, 4.6))
for i, (p, t) in enumerate([(pc, '"contain"'), (pg, '"grasp"')]):
    draw(fig2.add_subplot(1, 2, i+1, projection="3d"), p, t)
fig2.savefig("cam_karahi_feat/karahi_pred_contain_grasp.png", dpi=150, bbox_inches="tight")
print("saved karahi_pred_contain.png + karahi_pred_contain_grasp.png")
