"""Do SEEN-verb synonyms work? Query lean8 with paraphrases, compare AUPRC to the exact trained verb.
Held-out, clean human GT. CPU."""
import os,sys,json,hashlib
sys.path.insert(0,"src")
import numpy as np,torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN,AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper,VLMConfig
SCR="/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SYN={"grasp":["grip","grab"],"sit":["sit down"],"pour":["pour out"],"display":["show","present"],
     "contain":["hold"],"move":["push"],"lift":["pick up","raise"],"press":["tap"]}
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
    cg=torch.load(f"outputs/gnn_lean8_fold{k}.pt",map_location="cpu",weights_only=False)
    gc=AffordanceGNNConfig(**cg["cfg"]);m=AffordanceGNN(gc);m.load_state_dict(cg["model"]);m.eval();gnn[k]=(m,gc)
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu"))
allw=set(SYN)|{s for l in SYN.values() for s in l}
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
print(f"{'verb':9s} {'exact':>6s}  synonyms")
for v in SYN:
    ex=[];syn={s:[] for s in SYN[v]}
    for o in sorted(obj2fold):
        if o not in o2r or not clean(o,v):continue
        it,sel,_,_,_=_c.get(o,(None,)*5)
        h=human(o,v)
        if h is None:continue
        p0=pred(o,v);_,sel,_,_,_=_c[o];y=(h[sel]>=0.5).astype(int)
        if not(0<y.sum()<len(y)):continue
        ex.append(average_precision_score(y,p0))
        for s in SYN[v]:syn[s].append(average_precision_score(y,pred(o,s)))
    if ex:
        ss="  ".join(f"{s}={np.mean(syn[s]):.3f}(Δ{np.mean(syn[s])-np.mean(ex):+.2f})" for s in SYN[v])
        print(f"{v:9s} {np.mean(ex):6.3f}  {ss}")
