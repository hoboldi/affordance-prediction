"""Decompose the 0.921: how good is the PURE-GEAL model (gnn_pretrain_k24.pt, trained ONLY on GEAL
pseudolabels, zero manual labels) at predicting HUMAN labels? Eval on fold0-val clean human labels,
per verb + trained-mean AUPRC. Compare to GEAL+manualFT (gc_pre fold0 = 0.921) and MLP (0.838).
Caveat: pretrained saw these objects' GEAL pseudolabels (not their human labels) => transductive. CPU."""
import os, sys, json, collections, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
SUB = 24000; f = lambda t: t.float() if t is not None else None
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
cg = torch.load("outputs/gnn_pretrain_k24.pt", map_location="cpu", weights_only=False)
gcfg = AffordanceGNNConfig(**cg["cfg"]); gnn = AffordanceGNN(gcfg); gnn.load_state_dict(cg["model"]); gnn.eval()
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename="vertex_dino_fine.pt",
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
byv = collections.defaultdict(list)
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
            p = torch.sigmoid(gnn(emb[v], vlm_features=sl("vertex_features"), dino_vertex=sl("dino_vertex_features"),
                    slat_vertex=sl("slat_vertex_features"), vertex_normals=sl("vertex_normals"), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
        byv[v].append(average_precision_score(y, p))
GC_PRE = {"contain":0.989,"sit":0.937,"pour":0.862,"move":0.909,"display":0.981,"grasp":0.848}  # gc_pre fold0 (GEAL+manual FT)
print(f"{'verb':9s} {'pureGEAL':>9s} {'GEAL+manFT':>11s} {'manFT adds':>11s}")
pg, both = [], []
for v in TRAINED:
    m = float(np.mean(byv[v])) if byv[v] else float("nan"); pg.append(m); both.append(GC_PRE[v])
    print(f"{v:9s} {m:9.3f} {GC_PRE[v]:11.3f} {GC_PRE[v]-m:+11.3f}")
print(f"{'MEAN':9s} {np.mean(pg):9.3f} {np.mean(both):11.3f} {np.mean(both)-np.mean(pg):+11.3f}")
print(f"\nPure GEAL (0 manual labels) on human GT = {np.mean(pg):.3f}  ->  + manual fine-tuning = {np.mean(both):.3f}  (MLP full pipeline = 0.838)")
