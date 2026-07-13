"""HERO (clean, cherry-picked, NO top text): Human GT | MLP | GNN for 6 hand-picked (verb,object)
pairs with VISIBLE regions. Each object predicted with its HELD-OUT fold model (honest). Only small
column headers on the top row + a rotated verb label at left + a tiny AUPRC under each prediction."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SUB = 24000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)

PICKS = [  # (verb, object, view) — GNN clearly beats MLP; kept: bag, grasp cup, laptop
    ("grasp",   "handbag__30_1093_3177__v000",   (16, -60)),   # MLP 0.28 -> GNN 0.97  (+0.68)
    ("grasp",   "cup__14_158_900__v000",         (16, -60)),   # MLP 0.29 -> GNN 0.94  (+0.65)
    ("display", "laptop__122_14277_28317__v000", (25, 110)),   # MLP 0.63 -> GNN 1.00  (+0.37)
]

obj2fold = {}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: obj2fold[o] = k
gnn, mlp = {}, {}
for k in range(5):
    cg = torch.load(f"outputs/gnn_full_preclean_fold{k}.pt", map_location="cpu", weights_only=False)
    gc = AffordanceGNNConfig(**cg["cfg"]); m = AffordanceGNN(gc); m.load_state_dict(cg["model"]); m.eval(); gnn[k] = (m, gc)
    cm = torch.load(f"outputs/geomclean_fold{k}/last.pt", map_location="cpu", weights_only=False)
    mm = AffordanceMLP(mlp_head_config_from_model_cfg(cm["model_cfg"])); mm.load_state_dict(cm["model"]); mm.eval(); mlp[k] = mm
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))

def draw(ax, xyz, sel, c, gt, view):
    vmax = 1.0 if gt else float(max(0.05, c.max()))
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap="turbo", s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()

fig = plt.figure(figsize=(3 * 2.7, len(PICKS) * 2.55))
for ri, (v, o, view) in enumerate(PICKS):
    k = obj2fold[o]; it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
    knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gnn[k][1].knn_k)
    hb = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel]
    yb = (hb >= 0.5).astype(int); e = vlm.encode_text([v])[0]; sl = lambda kk: (it.get(kk).float()[idx] if it.get(kk) is not None else None)
    with torch.no_grad():
        pm = torch.sigmoid(mlp[k](e, slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"),
                dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"),
                vertex_positions=None, dino_vertex=sl("dino_vertex_features"), vertex_geom=sl("vertex_geom"))).numpy()
        pg = torch.sigmoid(gnn[k][0](e, vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
    am = average_precision_score(yb, pm); ag = average_precision_score(yb, pg)
    cells = [(hb, True, None), (pm, False, am), (pg, False, ag)]
    for ci, (fld, gt, ap) in enumerate(cells):
        ax = fig.add_subplot(len(PICKS), 3, ri * 3 + ci + 1, projection="3d")
        draw(ax, xyz, sel, fld, gt, view)
        if ri == 0:  # small column headers on the TOP ROW only
            ax.set_title(["Human GT", "MLP", "GNN (ours)"][ci], fontsize=12,
                         fontweight="bold", color=("black" if ci < 2 else "tab:green"), pad=-2)
        if ci == 0:
            ax.text2D(-0.08, 0.5, v, transform=ax.transAxes, fontsize=13, fontweight="bold", rotation=90, va="center")
        if ap is not None:  # tiny AUPRC at the BOTTOM of prediction cells
            ax.text2D(0.5, 0.02, f"AUPRC {ap:.2f}", transform=ax.transAxes, fontsize=8,
                      ha="center", color=("tab:green" if ci == 2 else "gray"))
plt.subplots_adjust(wspace=-0.1, hspace=0.02, left=0.04, right=0.99, top=0.97, bottom=0.01)
out = "outputs/renders/hero_gnn_vs_mlp_clean.png"; os.makedirs("outputs/renders", exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out, [(v, o.split("__")[0]) for v, o, _ in PICKS])
