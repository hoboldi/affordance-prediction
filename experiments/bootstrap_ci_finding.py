"""#4 Harden the two-class finding: bootstrap 95% CIs on the by-class ablation deltas.
Per (object,verb): AUPRC for lean8, nodino, nogeom. delta_dino = lean8-nodino, delta_geom = lean8-nogeom.
Bootstrap class-level mean deltas → do Geometric and Appearance CIs separate? CPU."""
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
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8],16)
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
CFG={"lean8":load("gnn_lean8"),"nodino":load("gnn_leanabl8_nodino"),"nogeom":load("gnn_leanabl8_nogeom")}
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
    D = z(gc.dino_vertex_dim) if cfgname=="nodino" else dino
    G = z(gc.geom_dim) if cfgname=="nogeom" else geom
    with torch.no_grad():
        p=torch.sigmoid(m(emb[v],vlm_features=z(gc.vlm_dim),dino_vertex=D,slat_vertex=z(gc.sam3d_dim),
            vertex_normals=z(gc.normals_dim),vertex_geom=G,knn_idx=knn)).numpy()
    h=human(o,v); y=(h[sel]>=0.5).astype(int)
    return average_precision_score(y,p[:,0] if p.ndim>1 else p)
rows=[]  # (verb, d_dino, d_geom)
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in V:
        h=human(o,v)
        if h is None or not clean(o,v): continue
        sel,idx,knn,k,_,_=feats(o); y=(h[sel]>=0.5).astype(int)
        if not (0<y.sum()<len(y)): continue
        a=ap("lean8",o,v); ad=ap("nodino",o,v); ag=ap("nogeom",o,v)
        rows.append((v, a-ad, a-ag))
rng=np.random.default_rng(0)
def boot(vals):
    vals=np.asarray(vals);
    if len(vals)==0: return (float("nan"),)*3
    bs=[vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(2000)]
    return vals.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)
print("=== Bootstrap 95% CI on by-class ablation deltas (n_items) ===")
for cls_name, cls in [("Geometric",GEO),("Appearance",APP)]:
    dd=[r[1] for r in rows if r[0] in cls]; dg=[r[2] for r in rows if r[0] in cls]
    m1,l1,h1=boot(dd); m2,l2,h2=boot(dg)
    print(f"\n{cls_name} (n={len(dd)}):")
    print(f"  drop-DINO delta:  {m1:+.3f}  [95% CI {l1:+.3f}, {h1:+.3f}]")
    print(f"  drop-geom delta:  {m2:+.3f}  [95% CI {l2:+.3f}, {h2:+.3f}]")
print("\nClaim check: Appearance drop-DINO CI should sit well above Geometric drop-DINO CI (DINO=appearance-specific);")
print("Geometric drop-geom CI should sit above Appearance drop-geom CI (geometry=geometric-specific).")
