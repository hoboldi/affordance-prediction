"""AIRTIGHT comparison: score the GNN (gnn_edgeconv_fold0) and the MLP (geomcv_fold0) on the
IDENTICAL vertex set per object, fold0-val, human GT. Deterministic per-object subsample (stable
across processes); the GNN's kNN is rebuilt on that exact subsample. Removes the full-vs-subsample
confound in the raw training logs. CPU."""
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
SUB = 30000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)  # stable across processes

# MLP (current best)
cm = torch.load("outputs/geomcv_fold0/last.pt", map_location="cpu", weights_only=False)
mlp = AffordanceMLP(mlp_head_config_from_model_cfg(cm["model_cfg"])); mlp.load_state_dict(cm["model"]); mlp.eval()
# GNN (edgeconv, best epoch)
cg = torch.load("outputs/gnn_edgeconv_fold0.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()

bcfg = mlp_head_config_from_model_cfg(cm["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]

byv = collections.defaultdict(lambda: {"mlp": [], "gnn": []})
for o in val:
    if o not in o2r: continue
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None:
            it = ds[o2r[o]]; V = len(it["vertex_positions"])
            r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(V, min(SUB, V), replace=False))
            idx = torch.as_tensor(sel, dtype=torch.long)
            knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
        yb = hb[sel]
        sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
        with torch.no_grad():
            pm = torch.sigmoid(mlp(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"),
                    dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"),
                    vertex_positions=None, dino_vertex=sl("dino_vertex_features"), vertex_geom=sl("vertex_geom"))).numpy()
            pg = torch.sigmoid(gnn(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                    slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"),
                    vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
        byv[v]["mlp"].append(average_precision_score(yb, pm))
        byv[v]["gnn"].append(average_precision_score(yb, pg))

print("=== fold0-val, IDENTICAL vertices per object (n objects per verb) ===")
print(f"{'verb':9s} {'MLP':>7s} {'GNN':>7s} {'delta':>7s}  n")
mm, gg = [], []
for v in TRAINED:
    m = np.mean(byv[v]["mlp"]); g = np.mean(byv[v]["gnn"]); mm.append(m); gg.append(g)
    print(f"{v:9s} {m:7.3f} {g:7.3f} {g-m:+7.3f}  {len(byv[v]['mlp'])}")
print(f"{'MEAN':9s} {np.mean(mm):7.3f} {np.mean(gg):7.3f} {np.mean(gg)-np.mean(mm):+7.3f}")
print(f"\nGNN spatial backbone vs MLP (same eval set): {np.mean(gg)-np.mean(mm):+.3f}")
