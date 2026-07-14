"""8-verb GEAL teacher comparison (numbers only): per-verb GEAL vs Ours (lean8 fold models, held-out, clean).
Ours = lean8 (DINO+geom+verb; clip/slat/normals zeroed). Verb-balanced macro. CPU."""
import os, sys, json, hashlib, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR="experiments/cv_harness"
DR="/home/datasets/customDatasets/cmr2/reconstructions"
V=["contain","pour","sit","move","display","grasp","press","lift"]
GEO=["contain","pour","sit","move","lift"]; APP=["display","grasp","press"]
SUB=24000
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8],16)
def geal(o,v):
    p=f"{DR}/{o}/vertex_pseudolabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v); return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
obj2fold={}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: obj2fold[o]=k
gnn={}
for k in range(5):
    cg=torch.load(f"outputs/gnn_lean8_fold{k}.pt",map_location="cpu",weights_only=False)
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
        sel=np.sort(r.choice(Vn,min(SUB,Vn),replace=False)); idx=torch.as_tensor(sel,dtype=torch.long)
        k=obj2fold[o]; knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],gnn[k][1].knn_k)
        _c[o]=(it,sel,idx,knn,k)
    it,sel,idx,knn,k=_c[o]; gc=gnn[k][1]
    sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None)
    z=lambda d:torch.zeros(len(sel),d)
    with torch.no_grad():
        p=torch.sigmoid(gnn[k][0](emb[v], vlm_features=z(gc.vlm_dim), dino_vertex=sl("dino_vertex_features"),
            slat_vertex=z(gc.sam3d_dim), vertex_normals=z(gc.normals_dim), vertex_geom=sl("vertex_geom"), knn_idx=knn)).numpy()
    return p,sel
byv={v:{"g":[],"o":[]} for v in V}
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in V:
        g=geal(o,v); h=human(o,v)
        if g is None or h is None or not clean(o,v): continue
        p,sel=predict(o,v); y=(h[sel]>=0.5).astype(int)
        if not (0<y.sum()<len(y)): continue
        byv[v]["g"].append(average_precision_score(y,g[sel])); byv[v]["o"].append(average_precision_score(y,p))
print("=== 8-verb GEAL vs Ours (held-out, clean) ===")
print(f"{'verb':9s} {'n':>3s} {'GEAL':>6s} {'Ours':>6s} {'Δ':>6s}")
gm,om=[],[]
for v in V:
    if byv[v]["g"]:
        gg,oo=np.mean(byv[v]["g"]),np.mean(byv[v]["o"]); gm.append(gg); om.append(oo)
        print(f"{v:9s} {len(byv[v]['g']):3d} {gg:6.3f} {oo:6.3f} {oo-gg:+6.3f}")
print(f"{'MACRO':9s}     {np.mean(gm):6.3f} {np.mean(om):6.3f} {np.mean(om)-np.mean(gm):+6.3f}")
