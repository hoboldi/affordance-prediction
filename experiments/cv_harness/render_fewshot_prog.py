"""Render the few-shot progression on ONE held-out object: press prediction at K=0,1,3,5 shots.
Fixed test object + fixed train pool (test never in train). Columns: Human GT | 0-shot | 1-shot | 3-shot |
5-shot, all on the same held-out laptop. GPU (fine-tune) then CPU (render)."""
import os, sys, glob, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
NOVEL = "press"; KS = [0, 1, 3, 5]; SUB = 10000; EPOCHS = 12; LR = 4e-4; PW = 5.0
dev = torch.device("cpu")   # GPUs busy with other users -> CPU; lighter SUB/EPOCHS to fit (qualitative fig)
base = torch.load("outputs/gnn_k24preclean_fold0.pt", map_location="cpu", weights_only=False)
cfg = AffordanceGNNConfig(**base["cfg"])
vlm = VLMWrapper(VLMConfig(device="cpu")); e = vlm.encode_text([NOVEL])[0].to(dev)
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]; vte = torch.stack([vlm.encode_text([v])[0] for v in TRAINED])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
objs = sorted({fp.split("/")[-2] for fp in glob.glob(f"human_gt_labels/*/vertex_manuallabels_{NOVEL}.pt")} & set(o2r))
laptops = [o for o in objs if o.split("__")[0] == "laptop"]
TEST = laptops[0]; POOL = [o for o in objs if o != TEST][:5]                 # test held out from train
print(f"test={TEST}  train pool={[o.split('__')[1] for o in POOL]}", flush=True)
cache = {}
def get(o):
    if o not in cache:
        it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
        r = np.random.default_rng(abs(hash(o)) % (2**32)); sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
        knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], cfg.knn_k).to(dev)
        ft = {k: (it.get(k).float()[idx].to(dev) if it.get(k) is not None else None)
              for k in ["vertex_features", "dino_vertex_features", "slat_vertex_features", "vertex_normals", "vertex_geom"]}
        cache[o] = (idx, knn, ft, xyz, sel)
    return cache[o]
def label(o, sel): return (np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{NOVEL}.pt", weights_only=False)).reshape(-1)[sel] >= 0.5).astype(int)
def fwd(m, o):
    idx, knn, ft, xyz, sel = get(o)
    return m(e, vlm_features=ft["vertex_features"], dino_vertex=ft["dino_vertex_features"], slat_vertex=ft["slat_vertex_features"],
             vertex_normals=ft["vertex_normals"], vertex_geom=ft["vertex_geom"], knn_idx=knn)
def train_k(K):
    m = AffordanceGNN(cfg, verb_text_embeddings=vte).to(dev); m.load_state_dict(base["model"])
    if K == 0: m.eval(); return m
    opt = torch.optim.Adam(m.parameters(), lr=LR); pw = torch.tensor(PW, device=dev)
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        for o in POOL[:K]:
            idx, knn, ft, xyz, sel = get(o)
            y = torch.as_tensor(label(o, sel), dtype=torch.float32, device=dev)
            Fn.binary_cross_entropy_with_logits(fwd(m, o), y, pos_weight=pw).backward()
        opt.step()
    m.eval(); return m
idx, knn, ft, xyz, sel = get(TEST); gt = label(TEST, sel)
preds = {}
for K in KS:
    m = train_k(K)
    with torch.no_grad(): p = torch.sigmoid(fwd(m, TEST)).float().cpu().numpy()
    preds[K] = (p, average_precision_score(gt, p)); print(f"K={K}: AUPRC={preds[K][1]:.2f}", flush=True)
binmap = ListedColormap(["#20124d", "#d7263d"])
cols = [("Human GT", gt, binmap, 1.0, "black")] + [(f"{K}-shot  AUPRC={preds[K][1]:.2f}", preds[K][0], "turbo", float(max(0.05, preds[K][0].max())), ("gray" if K == 0 else "tab:green")) for K in KS]
fig = plt.figure(figsize=(len(cols) * 2.5, 2.8))
for ci, (name, c, cmap, vmax, tc) in enumerate(cols):
    ax = fig.add_subplot(1, len(cols), ci + 1, projection="3d")
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=c, cmap=cmap, s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=40, azim=-70); ax.set_axis_off(); ax.set_title(name, fontsize=9, color=tc, fontweight="bold")
# (no suptitle — message lives in the poster/paper caption)
out = "outputs/renders/fewshot_progression.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
