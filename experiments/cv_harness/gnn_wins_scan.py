"""Rank clean held-out (obj,verb) by the GNN-MINUS-MLP gap (where the spatial GNN actually beats the
per-vertex MLP), require a VISIBLE region, and render a GT|MLP|GNN contact sheet to pick the hero rows.
Also dumps gnn_wins.json (gap-ranked). CPU."""
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
VERBS = ["contain", "sit", "pour", "move", "display", "grasp"]; SUB = 12000
f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)

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
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in VERBS}
_c = {}
def prep(o):
    if o not in _c:
        it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
        r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False))
        _c[o] = (it, xyz, sel, torch.as_tensor(sel, dtype=torch.long))
    return _c[o]
def raw(o, v): return np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)
def preds(o, v):
    k = obj2fold[o]; it, xyz, sel, idx = prep(o); e = emb[v]
    knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gnn[k][1].knn_k); sl = lambda kk: (it.get(kk).float()[idx] if it.get(kk) is not None else None)
    with torch.no_grad():
        pm = torch.sigmoid(mlp[k](e, slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"),
                dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"),
                vertex_positions=None, dino_vertex=sl("dino_vertex_features"), vertex_geom=sl("vertex_geom"))).numpy()
        pg = torch.sigmoid(gnn[k][0](e, vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
    return pm, pg
rows = []
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in VERBS:
        fp = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(fp): continue
        rr = raw(o, v)
        if float(((rr > 0.001) & (rr < 0.999)).mean()) >= 0.005: continue
        it, xyz, sel, idx = prep(o); y = (rr[sel] >= 0.5).astype(int); pf = float(y.mean())
        if not (0.01 < pf < 0.6): continue
        pm, pg = preds(o, v); am = float(average_precision_score(y, pm)); ag = float(average_precision_score(y, pg))
        rows.append({"obj": o, "cat": o.split("__")[0], "verb": v, "mlp": round(am, 3), "gnn": round(ag, 3), "gap": round(ag - am, 3), "posfrac": round(pf, 3)})
rows.sort(key=lambda d: -d["gap"])
json.dump(rows, open(f"{SCR}/gnn_wins.json", "w"), indent=1)
print("=== top GNN-minus-MLP gaps (visible regions only) ===")
for d in rows[:16]: print(f"  {d['verb']:8s} {d['cat']:8s} MLP={d['mlp']:.2f} GNN={d['gnn']:.2f} gap=+{d['gap']:.2f}  {d['obj'].split('__')[1]}")

# contact sheet of top-gap triplets: GT | MLP | GNN
K = 12; fig = plt.figure(figsize=(3 * 1.9, K * 1.75))
for ri, d in enumerate(rows[:K]):
    o, v = d["obj"], d["verb"]; it, xyz, sel, idx = prep(o); g = (raw(o, v)[sel] >= 0.5).astype(float); pm, pg = preds(o, v)
    for ci, (fld, gt, tag) in enumerate([(g, True, "GT"), (pm, False, f"MLP {d['mlp']:.2f}"), (pg, False, f"GNN {d['gnn']:.2f}")]):
        ax = fig.add_subplot(K, 3, ri * 3 + ci + 1, projection="3d")
        vmax = 1.0 if gt else float(max(0.05, fld.max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld, cmap="turbo", s=2, vmin=0, vmax=vmax)
        h = (xyz[sel].max(0)-xyz[sel].min(0)).max()/2+1e-6; mid = (xyz[sel].max(0)+xyz[sel].min(0))/2
        ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
        ax.view_init(elev=16, azim=-60); ax.set_axis_off()
        ax.set_title((f"{v}:{d['cat']} +{d['gap']:.2f}\n" if ci == 0 else "") + tag, fontsize=6)
out = f"{SCR}/gnn_wins_sheet.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); print("saved", out)
