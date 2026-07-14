"""Does reconstruction fidelity predict affordance performance? Per-object: Chamfer/F-score (vs CO3D GT)
AND ReVerb held-out AUPRC, split by verb class. Correlate overall + by class.
Scenario #1 (no corr): affordance robust to geometric infidelity -> justifies 2D->3D scaffold.
Scenario #2 (corr only for geometric verbs): two-class theory made an a-priori prediction that held.
Scenario #3 (corr across board): reconstruction is the bottleneck. CPU."""
import sys, os, json, glob, hashlib, itertools
sys.path.insert(0, "src")
import numpy as np, torch, trimesh
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "experiments/cv_harness"
SRC = "/home/datasets/customDatasets/cmr2/source"; DR = "/home/datasets/customDatasets/cmr2/reconstructions"
V = ["contain","pour","sit","move","display","grasp","press","lift"]
GEO = {"contain","pour","sit","move","lift"}; APP = {"grasp","display","press"}
NPTS = 12000; TAU = 0.05; SUB = 20000
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v); return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
obj2fold={}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: obj2fold[o]=k
def src_ply(s): p=s.split("__"); return f"{SRC}/{p[0]}/{p[1]}/pointcloud.ply" if len(p)>=2 else None
have=[s for s in sorted(obj2fold) if src_ply(s) and os.path.exists(src_ply(s)) and os.path.exists(f"{DR}/{s}/mesh.glb")]
print(f"objects with GT + mesh + in CV: {len(have)}", flush=True)
# ReVerb lean8 fold models
gnn={}
for k in range(5):
    cg=torch.load(f"outputs/gnn_lean8_fold{k}.pt",map_location="cpu",weights_only=False)
    gc=AffordanceGNNConfig(**cg["cfg"]); m=AffordanceGNN(gc); m.load_state_dict(cg["model"]); m.eval(); gnn[k]=(m,gc)
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu")); emb={v:vlm.encode_text([v])[0] for v in V}
# ---- fidelity helpers ----
def norm(P): P=P-P.mean(0); s=np.sqrt((P**2).sum(1)).mean(); return P/(s+1e-9)
def cham_f(A,B,tau=TAU):
    dab,_=cKDTree(B).query(A); dba,_=cKDTree(A).query(B)
    p=(dba<tau).mean(); r=(dab<tau).mean(); return float(dab.mean()+dba.mean()), float(2*p*r/(p+r+1e-9))
def oct_inits():
    out=[]
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1,-1],repeat=3):
            R=np.zeros((3,3))
            for i,p in enumerate(perm): R[i,p]=sg[i]
            if abs(np.linalg.det(R)-1)<1e-6: M=np.eye(4); M[:3,:3]=R; out.append(M)
    return out
INITS=oct_inits()
def fidelity(s):
    gt=trimesh.load(src_ply(s),process=False); G=np.asarray(gt.vertices,np.float64)
    if len(G)>NPTS: G=G[np.random.default_rng(0).choice(len(G),NPTS,replace=False)]
    mesh=trimesh.load(f"{DR}/{s}/mesh.glb",process=False,force="mesh"); M=np.asarray(mesh.sample(NPTS),np.float64)
    G=norm(G); M=norm(M); best=None
    for init in INITS:
        try: _,tr,_=trimesh.registration.icp(M,G,initial=init,max_iterations=30,threshold=1e-5)
        except Exception: continue
        c,f=cham_f(tr,G)
        if best is None or c<best[0]: best=(c,f)
    return best
# ---- affordance helper ----
_c={}
def auprc(s,v):
    if s not in _c:
        it=ds[o2r[s]]; Vn=len(it["vertex_positions"]); r=np.random.default_rng(seed_of(s))
        sel=np.sort(r.choice(Vn,min(SUB,Vn),replace=False)); idx=torch.as_tensor(sel,dtype=torch.long)
        k=obj2fold[s]; knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],gnn[k][1].knn_k)
        sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None); _c[s]=(sel,idx,knn,k,sl("dino_vertex_features"),sl("vertex_geom"))
    sel,idx,knn,k,dino,geom=_c[s]; m,gc=gnn[k]; z=lambda d:torch.zeros(len(sel),d)
    with torch.no_grad():
        p=torch.sigmoid(m(emb[v],vlm_features=z(gc.vlm_dim),dino_vertex=dino,slat_vertex=z(gc.sam3d_dim),
            vertex_normals=z(gc.normals_dim),vertex_geom=geom,knn_idx=knn)).numpy().reshape(-1)
    h=human(s,v); y=(h[sel]>=0.5).astype(int)
    return average_precision_score(y,p) if 0<y.sum()<len(y) else None
rows=[]
for i,s in enumerate(have):
    if s not in o2r: continue
    try: fid=fidelity(s)
    except Exception as e: print(f"  skip {s}: {e!r}",flush=True); continue
    if fid is None: continue
    aps={v:auprc(s,v) for v in V if clean(s,v)}; aps={v:a for v,a in aps.items() if a is not None}
    if not aps: continue
    geo=[aps[v] for v in aps if v in GEO]; app=[aps[v] for v in aps if v in APP]
    rows.append(dict(stem=s,cham=fid[0],f=fid[1],all=float(np.mean(list(aps.values()))),
                     geo=float(np.mean(geo)) if geo else None, app=float(np.mean(app)) if app else None))
    if (i+1)%20==0: print(f"  {i+1}/{len(have)}",flush=True)
json.dump(rows, open("experiments/fidelity_vs_perf.json","w"))
def corr(xs,ys,label):
    xs=np.array(xs); ys=np.array(ys)
    if len(xs)<4: print(f"  {label}: n={len(xs)} too few"); return
    pr,pp=pearsonr(xs,ys); sr,sp=spearmanr(xs,ys)
    print(f"  {label}: n={len(xs)}  Pearson r={pr:+.3f} (p={pp:.3f})  Spearman ρ={sr:+.3f} (p={sp:.3f})")
print(f"\n=== Fidelity (F-score) vs affordance AUPRC (n={len(rows)}) ===")
corr([r["f"] for r in rows], [r["all"] for r in rows], "ALL verbs   ")
gr=[r for r in rows if r["geo"] is not None]; ar=[r for r in rows if r["app"] is not None]
corr([r["f"] for r in gr], [r["geo"] for r in gr], "GEOMETRIC   ")
corr([r["f"] for r in ar], [r["app"] for r in ar], "APPEARANCE  ")
print("\nInterpretation: corr(GEOMETRIC) >> corr(APPEARANCE) => two-class theory predicts fidelity-dependence (scenario #2, strongest).")
print("near-zero everywhere => affordance robust to geometric infidelity (scenario #1). positive across board => reconstruction is the bottleneck (scenario #3).")
