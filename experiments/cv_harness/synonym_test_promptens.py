"""Test-time PROMPT ENSEMBLING (no retrain): does templating the held-out synonym query
recover it vs the bare word? Uses trained synaug8 fold models. CPU.
For each held-out synonym w: bare emb(w)  vs  mean of emb over context templates."""
import os,sys,json,hashlib
sys.path.insert(0,"src")
import numpy as np,torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN,AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper,VLMConfig
SCR="/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
HELD={"grasp":"seize","sit":"rest on","pour":"decant","display":"showcase","move":"reposition","press":"click","lift":"elevate","contain":"keep inside"}
# generic action/region templates (NOT hand-tuned per word) — standard CLIP prompt-ensembling
TPL=["{w}","you {w} this object","the part of the object you {w}","an action to {w} it","the region where you {w}"]
SUB=20000
def seed_of(o):return int(hashlib.md5(o.encode()).hexdigest()[:8],16)
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt";return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v);return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
obj2fold={}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]:obj2fold[o]=k
gnn={}
for k in range(5):
    cg=torch.load(f"outputs/gnn_synaug8_fold{k}.pt",map_location="cpu",weights_only=False)
    gc=AffordanceGNNConfig(**cg["cfg"]);m=AffordanceGNN(gc);m.load_state_dict(cg["model"]);m.eval();gnn[k]=(m,gc)
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu"))
def enc(w): return vlm.encode_text([w])[0]
def ensemble(w): return torch.stack([enc(t.format(w=w)) for t in TPL]).mean(0)
# precompute embeddings: exact verb, held bare, held ensembled
emb_exact={v:enc(v) for v in HELD}
emb_hbare={v:enc(HELD[v]) for v in HELD}
emb_hens ={v:ensemble(HELD[v]) for v in HELD}
_c={}
def pred(o,e):
    if o not in _c:
        it=ds[o2r[o]];Vn=len(it["vertex_positions"]);r=np.random.default_rng(seed_of(o))
        sel=np.sort(r.choice(Vn,min(SUB,Vn),replace=False));idx=torch.as_tensor(sel,dtype=torch.long)
        k=obj2fold[o];knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],gnn[k][1].knn_k);_c[o]=(it,sel,idx,knn,k)
    it,sel,idx,knn,k=_c[o];gc=gnn[k][1];sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None);z=lambda d:torch.zeros(len(sel),d)
    with torch.no_grad():
        return torch.sigmoid(gnn[k][0](e,vlm_features=z(gc.vlm_dim),dino_vertex=sl("dino_vertex_features"),
            slat_vertex=z(gc.sam3d_dim),vertex_normals=z(gc.normals_dim),vertex_geom=sl("vertex_geom"),knn_idx=knn)).numpy()
print("=== TEST-TIME PROMPT ENSEMBLING (synaug8, 5-fold, held-out synonyms) ===")
print(f"templates: {TPL}\n")
print(f"{'verb':9s} {'held-out':>11s} {'exact':>6s} {'bare':>6s} {'ENSEMBLE':>9s} {'Δens-bare':>10s}")
E,B,N=[],[],[]
for v in HELD:
    ex=[];ba=[];en=[]
    for o in sorted(obj2fold):
        if o not in o2r or not clean(o,v):continue
        h=human(o,v)
        if h is None:continue
        p0=pred(o,emb_exact[v]);_,sel,_,_,_=_c[o];y=(h[sel]>=0.5).astype(int)
        if not(0<y.sum()<len(y)):continue
        ex.append(average_precision_score(y,p0))
        ba.append(average_precision_score(y,pred(o,emb_hbare[v])))
        en.append(average_precision_score(y,pred(o,emb_hens[v])))
    if ex:
        E.append(np.mean(ex));B.append(np.mean(ba));N.append(np.mean(en))
        print(f"{v:9s} {HELD[v]:>11s} {np.mean(ex):6.3f} {np.mean(ba):6.3f} {np.mean(en):9.3f} {np.mean(en)-np.mean(ba):+10.3f}")
if E:
    print(f"\n{'MACRO':9s} {'':>11s} {np.mean(E):6.3f} {np.mean(B):6.3f} {np.mean(N):9.3f} {np.mean(N)-np.mean(B):+10.3f}")
    print(f"held-out drop vs exact:  bare {np.mean(B)-np.mean(E):+.3f}   ENSEMBLE {np.mean(N)-np.mean(E):+.3f}")
    print("(ENSEMBLE >> bare = context rescues held-out synonyms → prompt-ensembling helps open-vocab)")
