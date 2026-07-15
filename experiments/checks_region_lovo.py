"""P1 headline-protection checks.
CHECK 1 (region-size confound): is the appearance>geometric DINO-reliance an artifact of region size?
  Per (object,verb): DINO-drop = AUPRC(lean8)-AUPRC(nodino), positive-vertex fraction. Regress drop on
  class WITH and WITHOUT controlling for positive-fraction; also a size-matched subset.
CHECK 2 (leave-one-verb-out): does the appearance>geometric DINO gap survive dropping any single verb? CPU."""
import sys, os, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
from pathlib import Path
SCR = str(Path(__file__).resolve().parent / "cv_harness")  # absolute: DataRootDataset resolves relative paths against data_root
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
CFG={"lean8":load("gnn_lean8"),"nodino":load("gnn_leanabl8_nodino")}
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
        k=obj2fold[o]; knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],CFG["lean8"][k][1].knn_k)
        sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None)
        _c[o]=(sel,idx,knn,k,sl("dino_vertex_features"),sl("vertex_geom"))
    return _c[o]
def ap(cfgname,o,v):
    sel,idx,knn,k,dino,geom=feats(o); m,gc=CFG[cfgname][k]; z=lambda d:torch.zeros(len(sel),d)
    D=z(gc.dino_vertex_dim) if cfgname=="nodino" else dino
    with torch.no_grad():
        p=torch.sigmoid(m(emb[v],vlm_features=z(gc.vlm_dim),dino_vertex=D,slat_vertex=z(gc.sam3d_dim),
            vertex_normals=z(gc.normals_dim),vertex_geom=geom,knn_idx=knn)).numpy().reshape(-1)
    h=human(o,v); y=(h[sel]>=0.5).astype(int)
    return average_precision_score(y,p), float(y.mean())
rows=[]  # (verb, is_app, posfrac, drop)
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in V:
        if not clean(o,v): continue
        sel,idx,knn,k,_,_=feats(o); h=human(o,v); y=(h[sel]>=0.5).astype(int)
        if not (0<y.sum()<len(y)): continue
        al,pf=ap("lean8",o,v); an,_=ap("nodino",o,v)
        rows.append((v, v in APP, pf, al-an))
json.dump([list(r) for r in rows], open("experiments/region_lovo_rows.json","w"))
verb=np.array([r[0] for r in rows]); isapp=np.array([r[1] for r in rows],float)
pf=np.array([r[2] for r in rows]); drop=np.array([r[3] for r in rows])
print(f"n items = {len(rows)}")
print("\n=== CHECK 1: region size ===")
print("positive-vertex fraction by class:")
print(f"  Geometric : mean={pf[isapp==0].mean():.3f}  median={np.median(pf[isapp==0]):.3f}  IQR=[{np.percentile(pf[isapp==0],25):.3f},{np.percentile(pf[isapp==0],75):.3f}]")
print(f"  Appearance: mean={pf[isapp==1].mean():.3f}  median={np.median(pf[isapp==1]):.3f}  IQR=[{np.percentile(pf[isapp==1],25):.3f},{np.percentile(pf[isapp==1],75):.3f}]")
def ols(X,y):
    b,_,_,_=np.linalg.lstsq(X,y,rcond=None); return b
X_no=np.c_[np.ones(len(rows)),isapp]; X_sz=np.c_[np.ones(len(rows)),isapp,pf]
b_no=ols(X_no,drop); b_sz=ols(X_sz,drop)
print(f"\nDINO-drop ~ class (is_appearance coef): WITHOUT size control = {b_no[1]:+.3f}")
print(f"DINO-drop ~ class + posfrac  (is_appearance coef): WITH size control  = {b_sz[1]:+.3f}  (posfrac coef {b_sz[2]:+.3f})")
rng=np.random.default_rng(0); bs=[]
for _ in range(2000):
    i=rng.integers(0,len(rows),len(rows)); bs.append(ols(np.c_[np.ones(len(i)),isapp[i],pf[i]],drop[i])[1])
print(f"  bootstrap 95% CI on size-controlled class coef: [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
lo=max(np.percentile(pf[isapp==0],10),np.percentile(pf[isapp==1],10)); hi=min(np.percentile(pf[isapp==0],90),np.percentile(pf[isapp==1],90))
m=(pf>=lo)&(pf<=hi)
print(f"\nsize-matched subset posfrac in [{lo:.3f},{hi:.3f}] (n={m.sum()}):")
print(f"  Geometric DINO-drop = {drop[m&(isapp==0)].mean():+.3f} (n={ (m&(isapp==0)).sum() })   Appearance = {drop[m&(isapp==1)].mean():+.3f} (n={ (m&(isapp==1)).sum() })")
print("\n=== CHECK 2: leave-one-verb-out (DINO-drop by class, each verb removed) ===")
print(f"{'left out':10s} {'Geo drop':>9s} {'App drop':>9s} {'gap(App-Geo)':>13s}")
for lo_v in V:
    mm=verb!=lo_v; g=drop[mm&(isapp==0)]; a=drop[mm&(isapp==1)]
    if len(g) and len(a): print(f"{lo_v:10s} {g.mean():+9.3f} {a.mean():+9.3f} {a.mean()-g.mean():+13.3f}")
print("gap should stay clearly positive for every left-out verb if the DINO axis isn't carried by one verb.")
