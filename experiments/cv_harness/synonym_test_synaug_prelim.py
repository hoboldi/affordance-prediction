"""PRELIMINARY (folds 0,1,2 from snapshots): SEEN vs HELD-OUT synonyms on syn-aug model.
Firmer directional read than fold0-only, before folds 3-4 finish. CPU."""
import os,sys,json,hashlib
sys.path.insert(0,"src")
import numpy as np,torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN,AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper,VLMConfig
SCR="/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
FOLDS=[0,1,2]
SEEN={"grasp":"grip","sit":"sit down","pour":"pour out","display":"show","move":"slide","press":"tap","lift":"raise","contain":"store"}
HELD={"grasp":"seize","sit":"rest on","pour":"decant","display":"showcase","move":"reposition","press":"click","lift":"elevate","contain":"keep inside"}
SUB=20000
def seed_of(o):return int(hashlib.md5(o.encode()).hexdigest()[:8],16)
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt";return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v);return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
obj2fold={}
for k in FOLDS:
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]:obj2fold[o]=k
gnn={}
for k in FOLDS:
    cg=torch.load(f"/tmp/synaug8_fold{k}_snap.pt",map_location="cpu",weights_only=False)
    gc=AffordanceGNNConfig(**cg["cfg"]);m=AffordanceGNN(gc);m.load_state_dict(cg["model"]);m.eval();gnn[k]=(m,gc)
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu"))
allw=set(SEEN)|set(SEEN.values())|set(HELD.values())
emb={w:vlm.encode_text([w])[0] for w in allw}
_c={}
def pred(o,w):
    if o not in _c:
        it=ds[o2r[o]];Vn=len(it["vertex_positions"]);r=np.random.default_rng(seed_of(o))
        sel=np.sort(r.choice(Vn,min(SUB,Vn),replace=False));idx=torch.as_tensor(sel,dtype=torch.long)
        k=obj2fold[o];knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],gnn[k][1].knn_k);_c[o]=(it,sel,idx,knn,k)
    it,sel,idx,knn,k=_c[o];gc=gnn[k][1];sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None);z=lambda d:torch.zeros(len(sel),d)
    with torch.no_grad():
        return torch.sigmoid(gnn[k][0](emb[w],vlm_features=z(gc.vlm_dim),dino_vertex=sl("dino_vertex_features"),
            slat_vertex=z(gc.sam3d_dim),vertex_normals=z(gc.normals_dim),vertex_geom=sl("vertex_geom"),knn_idx=knn)).numpy()
nobj=len({o for o in obj2fold if o in o2r})
print(f"=== PRELIMINARY folds {FOLDS} ({nobj} held-out objects; fold2 snapshot ~ep17/18) ===")
print(f"{'verb':9s} {'exact':>6s} {'SEEN(trained)':>20s} {'HELD-OUT(novel)':>22s}")
E,S,H=[],[],[]
rows=[]
for v in SEEN:
    ex=[];se=[];he=[]
    for o in sorted(obj2fold):
        if o not in o2r or not clean(o,v):continue
        h=human(o,v)
        if h is None:continue
        p0=pred(o,v);_,sel,_,_,_=_c[o];y=(h[sel]>=0.5).astype(int)
        if not(0<y.sum()<len(y)):continue
        ex.append(average_precision_score(y,p0));se.append(average_precision_score(y,pred(o,SEEN[v])));he.append(average_precision_score(y,pred(o,HELD[v])))
    if ex:
        E.append(np.mean(ex));S.append(np.mean(se));H.append(np.mean(he))
        rows.append((v,len(ex),np.mean(ex),np.mean(se),np.mean(he)))
        print(f"{v:9s} {np.mean(ex):6.3f} {SEEN[v]+'='+format(np.mean(se),'.3f'):>20s} {HELD[v]+'='+format(np.mean(he),'.3f'):>22s}  (n={len(ex)})")
if E:
    print(f"\n{'MACRO':9s} {np.mean(E):6.3f} {'SEEN='+format(np.mean(S),'.3f'):>20s} {'HELD='+format(np.mean(H),'.3f'):>22s}")
    print(f"drop vs exact:   SEEN {np.mean(S)-np.mean(E):+.3f}   HELD-OUT {np.mean(H)-np.mean(E):+.3f}")
    # held-out split: which generalize (drop small) vs fail (drop big)
    gen=[r[0] for r in rows if r[2]-r[4]<0.20]; fail=[(r[0],HELD[r[0]]) for r in rows if r[2]-r[4]>=0.20]
    print(f"HELD-OUT generalize (<0.20 drop): {gen}")
    print(f"HELD-OUT fail (>=0.20 drop): {fail}")
