"""Visual: spatial GNN vs per-vertex MLP. Columns Human GT | MLP | GNN, rows = (object, verb) on
fold0-val. Shows both the accuracy gain and the COHERENCE (message passing -> less speckled maps).
Includes grasp (the one verb GNN loses) for honesty. Deterministic picks. CPU."""
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
PICKS = [("contain","bottle"),("sit","chair"),("pour","bottle"),("move","chair"),("display","laptop"),("grasp","handbag")]
SUB = 24000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
cm = torch.load(sys.argv[3] if len(sys.argv)>3 else "outputs/geomcv_fold0/last.pt", map_location="cpu", weights_only=False)
mlp = AffordanceMLP(mlp_head_config_from_model_cfg(cm["model_cfg"])); mlp.load_state_dict(cm["model"]); mlp.eval()
cg = torch.load(sys.argv[1] if len(sys.argv)>1 else "outputs/gnn_edgeconv_fold0.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
def has(o, v): return os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")
def first_obj(cat, v): return next((o for o in sorted(val) if o.split("__")[0] == cat and has(o, v)), None)

rows = [(v, first_obj(cat, v)) for v, cat in PICKS]
rows = [(v, o) for v, o in rows if o is not None]
fig = plt.figure(figsize=(3 * 2.7, len(rows) * 2.7))
for ri, (v, o) in enumerate(rows):
    it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
    r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
    knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
    hb = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)[sel]
    yb = (hb >= 0.5).astype(int); e = vlm.encode_text([v])[0]
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        pm = torch.sigmoid(mlp(e, slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"),
                dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"),
                vertex_positions=None, dino_vertex=sl("dino_vertex_features"), vertex_geom=sl("vertex_geom"))).numpy()
        pg = torch.sigmoid(gnn(e, vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"),
                vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
    am = average_precision_score(yb, pm); ag = average_precision_score(yb, pg)
    cols = [("Human GT", hb, None, "black"), (f"MLP  AUPRC={am:.2f}", pm, am, "gray"), (f"GNN  AUPRC={ag:.2f}", pg, ag, "tab:green")]
    for ci, (name, fld, ap, tcol) in enumerate(cols):
        ax = fig.add_subplot(len(rows), 3, ri * 3 + ci + 1, projection="3d")
        vmax = 1.0 if ci == 0 else float(max(0.05, fld.max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld, cmap="turbo", s=3, vmin=0, vmax=vmax)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
        ax.view_init(elev=16, azim=-60); ax.set_axis_off()
        ax.set_title(name, fontsize=9, color=tcol, fontweight=("bold" if ci == 2 else "normal"))
        if ci == 0: ax.text2D(-0.15, 0.5, f'"{v}"\n{o.split("__")[0]}', transform=ax.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
fig.suptitle("Spatial GNN vs per-vertex MLP (fold0-val, held-out) — Human GT | MLP | GNN\n"
             "CLEAN data (contamination-free) — previous best (MLP full-FT+geom) vs new best (GEAL-pretrained GNN). Both trained on clean hand-labels only.", fontsize=10, y=0.998)
os.makedirs("outputs/renders", exist_ok=True)
OUT=sys.argv[2] if len(sys.argv)>2 else "outputs/renders/gnn_vs_mlp.png"
fig.savefig(OUT, dpi=125, bbox_inches="tight")
print("saved", OUT, [(v, o.split("__")[0]) for v, o in rows])
