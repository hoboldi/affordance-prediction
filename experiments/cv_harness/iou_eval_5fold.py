"""5-fold eval with three metrics + threshold calibration. Per verb, MLP vs k24-GNN:
  AUPRC (ranking) | IoU@0.5 (raw fixed thresh) | IoU@tau* (per-verb threshold CALIBRATED leave-one-fold-out)
  | best-IoU (per-object oracle threshold = ceiling).
Uses whatever gnn_k24_fold{k}.pt exist (auto-detect); with 1 fold, calibration falls back to same-fold optimal
(flagged). Shows the GNN's 'flooding' is a calibration issue: raw-0.5 floods (grasp/sit), tau* recovers it. CPU."""
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
SUB = 20000; THRS = np.linspace(0.05, 0.95, 19); f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def iou_arr(p, y, thr):
    pb = p >= thr; yb = y == 1; u = (pb | yb).sum()
    return float((pb & yb).sum() / u) if u > 0 else np.nan
FOLDS = [k for k in range(5) if os.path.exists(f"outputs/gnn_k24_fold{k}.pt") and os.path.exists(f"outputs/geomcv_fold{k}/last.pt")]
print(f"folds available: {FOLDS}")

vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}

# per (fold,verb): list of dicts {mlp_p, gnn_p, y} ; scalar metrics accumulated too
store = collections.defaultdict(lambda: collections.defaultdict(list))   # store[fold][verb] = [ (gnn_p,y), ... ]
scal = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))  # scal[metric][model][verb]
for k in FOLDS:
    cm = torch.load(f"outputs/geomcv_fold{k}/last.pt", map_location="cpu", weights_only=False)
    mlp = AffordanceMLP(mlp_head_config_from_model_cfg(cm["model_cfg"])); mlp.load_state_dict(cm["model"]); mlp.eval()
    cg = torch.load(f"outputs/gnn_k24_fold{k}.pt", map_location="cpu", weights_only=False)
    gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
    val = json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]
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
            y = hb[sel]; sl = lambda kk: (it.get(kk).float()[idx] if it.get(kk) is not None else None)
            with torch.no_grad():
                pm = torch.sigmoid(mlp(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"),
                        dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"),
                        vertex_positions=None, dino_vertex=sl("dino_vertex_features"), vertex_geom=sl("vertex_geom"))).numpy()
                pg = torch.sigmoid(gnn(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                        slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
            store[k][v].append((pg, y))
            scal["auprc"]["MLP"][v].append(average_precision_score(y, pm)); scal["auprc"]["GNN"][v].append(average_precision_score(y, pg))
            scal["iou50"]["MLP"][v].append(iou_arr(pm, y, 0.5)); scal["iou50"]["GNN"][v].append(iou_arr(pg, y, 0.5))
            scal["biou"]["MLP"][v].append(max(iou_arr(pm, y, t) for t in THRS)); scal["biou"]["GNN"][v].append(max(iou_arr(pg, y, t) for t in THRS))

# calibrated GNN IoU: per verb, tau* fit on OTHER folds (LOFO), applied to held-out fold (fallback: same fold)
cal = collections.defaultdict(list)
for v in TRAINED:
    for k in FOLDS:
        others = [kk for kk in FOLDS if kk != k] or [k]
        def mean_iou_at(t, folds):
            xs = [iou_arr(pg, y, t) for kk in folds for (pg, y) in store[kk][v]]
            return np.nanmean(xs) if xs else np.nan
        taus = [(t, mean_iou_at(t, others)) for t in THRS]
        tau = max(taus, key=lambda z: (z[1] if not np.isnan(z[1]) else -1))[0]
        for (pg, y) in store[k][v]: cal[v].append(iou_arr(pg, y, tau))

def mrow(d, v): return np.nanmean(d[v]) if d[v] else np.nan
print(f"\n{'verb':9s} |  AUPRC MLP/GNN  | IoU@0.5 MLP/GNN | IoU@tau* GNN(cal) | best-IoU MLP/GNN")
A = collections.defaultdict(lambda: collections.defaultdict(list))
for v in TRAINED:
    a = (mrow(scal['auprc']['MLP'], v), mrow(scal['auprc']['GNN'], v))
    i5 = (mrow(scal['iou50']['MLP'], v), mrow(scal['iou50']['GNN'], v))
    ic = np.nanmean(cal[v]); bi = (mrow(scal['biou']['MLP'], v), mrow(scal['biou']['GNN'], v))
    for nm, val_ in [('aM',a[0]),('aG',a[1]),('i5M',i5[0]),('i5G',i5[1]),('ic',ic),('bM',bi[0]),('bG',bi[1])]: A['x'][nm].append(val_)
    print(f"{v:9s} |  {a[0]:.3f}  {a[1]:.3f}  |  {i5[0]:.3f}  {i5[1]:.3f}  |     {ic:.3f}      |  {bi[0]:.3f}  {bi[1]:.3f}")
m = {nm: np.nanmean(A['x'][nm]) for nm in A['x']}
print(f"{'MEAN':9s} |  {m['aM']:.3f}  {m['aG']:.3f}  |  {m['i5M']:.3f}  {m['i5G']:.3f}  |     {m['ic']:.3f}      |  {m['bM']:.3f}  {m['bG']:.3f}")
print(f"\nGNN raw-0.5 IoU {m['i5G']:.3f} -> calibrated {m['ic']:.3f} (fix the flooding) ; ceiling best-IoU {m['bG']:.3f}. MLP IoU@0.5 {m['i5M']:.3f}.")
