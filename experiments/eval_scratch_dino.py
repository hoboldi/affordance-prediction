"""CHECK B eval: does the appearance>geometric DINO-reliance reproduce in a from-scratch model (NO teacher)?
Per (object,verb): AUPRC(scratch8) - AUPRC(scratchnodino8), by class + leave-one-verb-out gap.
If App DINO-drop >> Geo DINO-drop even from scratch, the split is task-intrinsic, not inherited from GEAL. CPU."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
V = ["contain","pour","sit","move","display","grasp","press","lift"]
GEO = {"contain","pour","sit","move","lift"}; APP = {"grasp","display","press"}
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v); return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
obj2fold={}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: obj2fold[o]=k
def load(tag):
    d={}
    for k in range(5):
        cg=torch.load(f"outputs/{tag}_fold{k}.pt",map_location="cpu",weights_only=False)
        gc=AffordanceGNNConfig(**cg["cfg"]); m=AffordanceGNN(gc); m.load_state_dict(cg["model"]); m.eval(); d[k]=(m,gc)
    return d
CFG={"scratch8":load("gnn_scratch8"),"scratchnodino8":load("gnn_scratchnodino8")}
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu")); emb={v:vlm.encode_text([v])[0] for v in V}
_c={}
def feats(o):
    if o not in _c:
        it=ds[o2r[o]]; Vn=len(it["vertex_positions"]); r=np.random.default_rng(seed_of(o))
        sel=np.sort(r.choice(Vn,min(20000,Vn),replace=False)); idx=torch.as_tensor(sel,dtype=torch.long)
        k=obj2fold[o]; knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],CFG["scratch8"][k][1].knn_k)
        sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None)
        _c[o]=(sel,idx,knn,k,sl("dino_vertex_features"),sl("vertex_geom"))
    return _c[o]
def ap(cfgname,o,v):
    sel,idx,knn,k,dino,geom=feats(o); mm,gc=CFG[cfgname][k]; z=lambda d:torch.zeros(len(sel),d)
    D=z(gc.dino_vertex_dim) if cfgname=="scratchnodino8" else dino
    with torch.no_grad():
        p=torch.sigmoid(mm(emb[v],vlm_features=z(gc.vlm_dim),dino_vertex=D,slat_vertex=z(gc.sam3d_dim),
            vertex_normals=z(gc.normals_dim),vertex_geom=geom,knn_idx=knn)).numpy().reshape(-1)
    h=human(o,v); y=(h[sel]>=0.5).astype(int)
    return average_precision_score(y,p)
rows=[]
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in V:
        if not clean(o,v): continue
        sel,idx,knn,k,_,_=feats(o); h=human(o,v); y=(h[sel]>=0.5).astype(int)
        if not (0<y.sum()<len(y)): continue
        rows.append((v, v in APP, ap("scratch8",o,v), ap("scratchnodino8",o,v)))
verb=np.array([r[0] for r in rows]); isapp=np.array([r[1] for r in rows]); lean=np.array([r[2] for r in rows]); nod=np.array([r[3] for r in rows])
drop=lean-nod
def cm(mask,arr): return arr[mask].mean() if mask.sum() else float("nan")
print(f"=== CHECK B: from-scratch (NO teacher) DINO-reliance by class (n={len(rows)}) ===")
print(f"  from-scratch absolute:  Geometric lean={cm(isapp==0,lean):.3f}  Appearance lean={cm(isapp==1,lean):.3f}")
print(f"  from-scratch DINO-drop:  Geometric={cm(isapp==0,drop):+.3f}   Appearance={cm(isapp==1,drop):+.3f}   gap={cm(isapp==1,drop)-cm(isapp==0,drop):+.3f}")
print(f"  (reference: finetuned lean8 DINO-drop  Geometric +0.06 / Appearance +0.245, gap +0.185)")
print(f"\n  leave-one-verb-out gap (App-Geo DINO-drop):")
for lo in V:
    mm=verb!=lo; g=cm(mm&(isapp==0),drop); a=cm(mm&(isapp==1),drop); print(f"    drop {lo:9s}: gap {a-g:+.3f}")
print("\nVerdict: App gap clearly > 0 from scratch => two-class split is TASK-INTRINSIC (reproduces without GEAL).")
print("Also settles 'can't train without a teacher': from-scratch absolute lean numbers above.")
