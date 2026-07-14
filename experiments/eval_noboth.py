"""Measured drop-both (DINO+geom removed) cell: 5-fold held-out clean macro AUPRC for gnn_leanabl8_noboth.
Zeros clip/slat/normals AND dino/geom (verb + spatial structure only). CPU."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "experiments/cv_harness"
V = ["contain","pour","sit","move","display","grasp","press","lift"]
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8],16)
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v); return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
obj2fold={}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: obj2fold[o]=k
gnn={}
for k in range(5):
    cg=torch.load(f"outputs/gnn_leanabl8_noboth_fold{k}.pt",map_location="cpu",weights_only=False)
    gc=AffordanceGNNConfig(**cg["cfg"]); m=AffordanceGNN(gc); m.load_state_dict(cg["model"]); m.eval(); gnn[k]=(m,gc)
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu")); emb={v:vlm.encode_text([v])[0] for v in V}
_c={}
def predict(o,v):
    if o not in _c:
        it=ds[o2r[o]]; Vn=len(it["vertex_positions"]); r=np.random.default_rng(seed_of(o))
        sel=np.sort(r.choice(Vn,min(24000,Vn),replace=False)); idx=torch.as_tensor(sel,dtype=torch.long)
        k=obj2fold[o]; knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],gnn[k][1].knn_k); _c[o]=(it,sel,idx,knn,k)
    it,sel,idx,knn,k=_c[o]; gc=gnn[k][1]; z=lambda d:torch.zeros(len(sel),d)
    with torch.no_grad():  # ALL per-vertex features zeroed (drop-both): only verb + kNN structure
        p=torch.sigmoid(gnn[k][0](emb[v], vlm_features=z(gc.vlm_dim), dino_vertex=z(gc.dino_vertex_dim),
            slat_vertex=z(gc.sam3d_dim), vertex_normals=z(gc.normals_dim), vertex_geom=z(gc.geom_dim), knn_idx=knn)).numpy()
    return p,sel
byv={v:[] for v in V}
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in V:
        h=human(o,v)
        if h is None or not clean(o,v): continue
        p,sel=predict(o,v); y=(h[sel]>=0.5).astype(int)
        if not (0<y.sum()<len(y)): continue
        byv[v].append(average_precision_score(y,p[:,0] if p.ndim>1 else p))
print("=== noboth (DINO+geom removed) — 5-fold held-out clean macro AUPRC ===")
ms=[]
for v in V:
    if byv[v]: ms.append(np.mean(byv[v])); print(f"  {v:9s} n={len(byv[v]):3d}  {np.mean(byv[v]):.3f}")
print(f"  MACRO (drop-both cell): {np.mean(ms):.3f}   (chance ~0.20; lean8 = 0.863)")
