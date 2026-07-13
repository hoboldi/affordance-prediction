"""Why does the 'flooded'-looking k24 grasp map on the handbag score HIGHER (0.96) than the tidier MLP (0.88)?
AUPRC measures RANKING (are positives ranked above negatives), not how much of the mesh lights up. Measure it:
score distributions on positive (grasp) vs negative vertices, for MLP vs k24-GNN, on the exact render object. CPU."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
V0 = "grasp"; SUB = 24000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
cm = torch.load("outputs/geomcv_fold0/last.pt", map_location="cpu", weights_only=False)
mlp = AffordanceMLP(mlp_head_config_from_model_cfg(cm["model_cfg"])); mlp.load_state_dict(cm["model"]); mlp.eval()
cg = torch.load("outputs/gnn_k24_fold0.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); e = vlm.encode_text([V0])[0]
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
o = next(x for x in sorted(val) if x.split("__")[0] == "handbag" and os.path.exists(f"human_gt_labels/{x}/vertex_manuallabels_{V0}.pt"))
print(f"object: {o}")
it = ds[o2r[o]]; Vn = len(it["vertex_positions"])
r = np.random.default_rng(seed_of(o)); sel = np.sort(r.choice(Vn, min(SUB, Vn), replace=False)); idx = torch.as_tensor(sel, dtype=torch.long)
knn = AffordanceGNN._build_knn(it["vertex_positions"].float()[idx], gcfg.knn_k)
hb = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{V0}.pt", weights_only=False)).reshape(-1)[sel]
y = (hb >= 0.5).astype(int)
sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
with torch.no_grad():
    pm = torch.sigmoid(mlp(e, slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"),
            dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"),
            vertex_positions=None, dino_vertex=sl("dino_vertex_features"), vertex_geom=sl("vertex_geom"))).numpy()
    pg = torch.sigmoid(gnn(e, vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"),
            vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()

print(f"positives (grasp) = {y.sum()}/{len(y)} = {y.mean():.3f} base rate\n")
for name, p in [("MLP", pm), ("k24-GNN", pg)]:
    pos, neg = p[y == 1], p[y == 0]
    npos = int(y.sum())
    topk = np.argsort(-p)[:npos]                       # the npos highest-scored vertices
    prec_at_npos = y[topk].mean()                      # of the top-npos predicted, fraction truly grasp
    print(f"--- {name} ---  AUPRC={average_precision_score(y,p):.3f}  ROC-AUC={roc_auc_score(y,p):.3f}")
    print(f"    score on POS: mean={pos.mean():.3f} median={np.median(pos):.3f}  |  on NEG: mean={neg.mean():.3f} median={np.median(neg):.3f}")
    print(f"    NEG fraction > 0.5 (visual 'flooding'): {(neg>0.5).mean():.3f}   |  POS fraction > 0.5: {(pos>0.5).mean():.3f}")
    print(f"    precision@{npos} (top-scored = truly grasp): {prec_at_npos:.3f}")
    print(f"    POS ranked above NEG median? pos_median {np.median(pos):.3f} vs neg_p95 {np.percentile(neg,95):.3f}\n")
print("Key: AUPRC/precision@k reward the POS vertices being ranked at the TOP. 'Flooding' = many NEG with moderate")
print("scores; it only hurts AUPRC if those NEG outrank true POS. Compare precision@npos and pos-vs-neg separation.")
