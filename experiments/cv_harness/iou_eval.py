"""Quantify the 'flooding': AUPRC (ranking) vs IoU@0.5 (fixed-threshold deployment) vs best-IoU (optimal
threshold), MLP vs k24-GNN, per verb, fold0-val, identical vertices. If GNN floods, IoU@0.5 should show
it (false positives at 0.5) even where AUPRC wins; best-IoU tells whether the map is thresholdable at all. CPU."""
import os, sys, json, collections, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
SUB = 24000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def iou(p, y, thr):
    pb = p >= thr; yb = y == 1; u = (pb | yb).sum()
    return float((pb & yb).sum() / u) if u > 0 else float("nan")
def best_iou(p, y): return max(iou(p, y, t) for t in np.linspace(0.05, 0.95, 19))

cm = torch.load("outputs/geomcv_fold0/last.pt", map_location="cpu", weights_only=False)
mlp = AffordanceMLP(mlp_head_config_from_model_cfg(cm["model_cfg"])); mlp.load_state_dict(cm["model"]); mlp.eval()
cg = torch.load("outputs/gnn_k24_fold0.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
R = {m: collections.defaultdict(lambda: collections.defaultdict(list)) for m in ["MLP", "GNN"]}
for o in val:
    if o not in o2r: continue
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None:
            it = ds[o2r[o]]; Vn = len(it["vertex_positions"])
            rr = np.random.default_rng(seed_of(o)); sel = np.sort(rr.choice(Vn, min(SUB, Vn), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
            knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
        y = hb[sel]; sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
        with torch.no_grad():
            pm = torch.sigmoid(mlp(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"),
                    dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"),
                    vertex_positions=None, dino_vertex=sl("dino_vertex_features"), vertex_geom=sl("vertex_geom"))).numpy()
            pg = torch.sigmoid(gnn(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                    slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
        for m, p in [("MLP", pm), ("GNN", pg)]:
            R[m][v]["auprc"].append(average_precision_score(y, p))
            R[m][v]["iou50"].append(iou(p, y, 0.5)); R[m][v]["biou"].append(best_iou(p, y))

print(f"{'verb':9s} | {'AUPRC (rank)':>16s} | {'IoU@0.5 (deploy)':>18s} | {'best-IoU':>16s}")
print(f"{'':9s} | {'MLP':>7s} {'GNN':>7s} | {'MLP':>8s} {'GNN':>8s} | {'MLP':>7s} {'GNN':>7s}")
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for v in TRAINED:
    r = {m: {k: np.nanmean(R[m][v][k]) for k in ["auprc", "iou50", "biou"]} for m in ["MLP", "GNN"]}
    for m in ["MLP", "GNN"]:
        for k in ["auprc", "iou50", "biou"]: agg[m][k].append(r[m][k])
    print(f"{v:9s} | {r['MLP']['auprc']:7.3f} {r['GNN']['auprc']:7.3f} | {r['MLP']['iou50']:8.3f} {r['GNN']['iou50']:8.3f} | {r['MLP']['biou']:7.3f} {r['GNN']['biou']:7.3f}")
print(f"{'MEAN':9s} | " + " ".join(f"{np.mean(agg[m]['auprc']):7.3f}" for m in ["MLP","GNN"]) +
      " | " + " ".join(f"{np.mean(agg[m]['iou50']):8.3f}" for m in ["MLP","GNN"]) +
      " | " + " ".join(f"{np.mean(agg[m]['biou']):7.3f}" for m in ["MLP","GNN"]))
print("\nIoU@0.5: does the fixed-threshold binary map suffer from GNN flooding? best-IoU: is the map thresholdable at all?")
