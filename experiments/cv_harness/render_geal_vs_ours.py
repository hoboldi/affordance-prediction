"""GEAL comparison (clean, held-out): Human GT | GEAL teacher | Ours (GNN). GEAL = the pseudolabels we
distilled FROM (vertex_pseudolabels_<verb>.pt). Ours predicted by the held-out fold model. Also scans ALL
val objects for the per-verb GEAL-vs-Ours AUPRC table (the honest '+0.058', broken down). Text-free figure."""
import os, sys, json, hashlib, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
DR = "/home/datasets/customDatasets/cmr2/reconstructions"
VERBS = ["contain","sit","pour","move","display","grasp"]
SUB = 24000
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def geal(o, v):
    p = f"{DR}/{o}/vertex_pseudolabels_{v}.pt"
    return np.asarray(torch.load(p, weights_only=False)).reshape(-1) if os.path.exists(p) else None
def human(o, v):
    p = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
    return np.asarray(torch.load(p, weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o, v):  # skip GEAL-contaminated human labels (continuous)
    h = human(o, v); return h is not None and ((h > 0.001) & (h < 0.999)).mean() < 0.005

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
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in VERBS}
_pc = {}
def predict(o, v):  # held-out Ours prediction on the SUB subsample
    if o not in _pc:
        it = ds[o2r[o]]; V = len(it["vertex_positions"]); r = np.random.default_rng(seed_of(o))
        sel = np.sort(r.choice(V, min(SUB, V), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
        k = obj2fold[o]; knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gnn[k][1].knn_k)
        _pc[o] = (it, sel, idx, knn, k)
    it, sel, idx, knn, k = _pc[o]; sl = lambda kk: (it.get(kk).float()[idx] if it.get(kk) is not None else None)
    with torch.no_grad():
        p = torch.sigmoid(gnn[k][0](emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
              slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
    return p, sel

# ---- SCAN all val objects: per-verb GEAL vs Ours AUPRC (honest table) ----
byv = collections.defaultdict(lambda: {"geal": [], "ours": []}); cand = []
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in VERBS:
        g = geal(o, v); h = human(o, v)
        if g is None or h is None or not clean(o, v): continue
        p, sel = predict(o, v); y = (h[sel] >= 0.5).astype(int)
        if not (0 < y.sum() < len(y)): continue
        ga = average_precision_score(y, g[sel]); oa = average_precision_score(y, p)
        byv[v]["geal"].append(ga); byv[v]["ours"].append(oa)
        cand.append((v, o, ga, oa, oa - ga, y.sum() / len(y)))
print("=== per-verb GEAL vs Ours (held-out val, clean) ===")
tot_g, tot_o = [], []
for v in VERBS:
    if byv[v]["geal"]:
        g, o_ = np.mean(byv[v]["geal"]), np.mean(byv[v]["ours"])
        tot_g += byv[v]["geal"]; tot_o += byv[v]["ours"]
        print(f"  {v:8s} n={len(byv[v]['geal']):3d}  GEAL={g:.3f}  Ours={o_:.3f}  Δ={o_-g:+.3f}")
print(f"  {'ALL':8s} n={len(tot_g):3d}  GEAL={np.mean(tot_g):.3f}  Ours={np.mean(tot_o):.3f}  Δ={np.mean(tot_o)-np.mean(tot_g):+.3f}")

# ---- RENDER: pick diverse high-contrast, visible examples ----
def draw(ax, xyz, sel, c, gt):
    vmax = 1.0 if gt else float(max(0.05, c.max()))
    ax.scatter(xyz[sel,0], xyz[sel,2], xyz[sel,1], c=c, cmap="turbo", s=3, vmin=0, vmax=vmax)
    h = (xyz[sel].max(0)-xyz[sel].min(0)).max()/2+1e-6; mid=(xyz[sel].max(0)+xyz[sel].min(0))/2
    ax.set_xlim(mid[0]-h,mid[0]+h); ax.set_ylim(mid[2]-h,mid[2]+h); ax.set_zlim(mid[1]-h,mid[1]+h)
    ax.view_init(elev=16, azim=-60); ax.set_axis_off()
cand = [c for c in cand if 0.05 < c[5] < 0.6 and c[3] > 0.7 and c[2] < 0.7]  # ours good, GEAL weaker, mid-size region
cand.sort(key=lambda c: -c[4])
picks, seen = [], set()
for v, o, ga, oa, d, fr in cand:
    key = (v, o.split("__")[0])
    if key in seen: continue
    seen.add(key); picks.append((v, o, ga, oa))
    if len(picks) == 4: break
fig = plt.figure(figsize=(3*2.7, len(picks)*2.55))
for ri,(v,o,ga,oa) in enumerate(picks):
    it = _pc[o][0]; xyz = it["vertex_positions"].numpy(); p, sel = predict(o, v)
    hb = human(o, v)[sel]; g = geal(o, v)[sel]
    for ci,(fld,gt,ap) in enumerate([(hb,True,None),(g,False,ga),(p,False,oa)]):
        ax = fig.add_subplot(len(picks),3,ri*3+ci+1,projection="3d"); draw(ax,xyz,sel,fld,gt)
        if ri==0: ax.set_title(["Human GT","GEAL (teacher)","Ours"][ci],fontsize=12,fontweight="bold",
                               color=("black" if ci<2 else "tab:green"),pad=-2)
        if ci==0: ax.text2D(-0.08,0.5,v,transform=ax.transAxes,fontsize=13,fontweight="bold",rotation=90,va="center")
        if ap is not None: ax.text2D(0.5,0.02,f"AUPRC {ap:.2f}",transform=ax.transAxes,fontsize=8,ha="center",
                                     color=("tab:green" if ci==2 else "gray"))
plt.subplots_adjust(wspace=-0.1,hspace=0.02,left=0.04,right=0.99,top=0.97,bottom=0.01)
out="final_figures/result_geal_vs_ours.png"; fig.savefig(out,dpi=140,bbox_inches="tight")
print("saved",out,"picks:",[(v,o.split('__')[0],round(ga,2),round(oa,2)) for v,o,ga,oa in picks])
