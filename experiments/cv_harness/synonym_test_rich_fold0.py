"""Compare RICH-augmented model vs baseline syn-aug on held-out synonyms (fold0 only, apples-to-apples).
Both warm-started from same pretrain, both fold0; only diff = augmentation richness. CPU.
Columns: exact | syn-aug bare | RICH bare | RICH template-ensemble."""
import os,sys,json,hashlib
sys.path.insert(0,"src")
import numpy as np,torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN,AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper,VLMConfig
SCR="/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
HELD={"grasp":"seize","sit":"rest on","pour":"decant","display":"showcase","move":"reposition","press":"click","lift":"elevate","contain":"keep inside"}
TPL=["{w}","you {w} this object","the part where you {w}","an action to {w}"]  # same templates used in rich training
SUB=20000
def seed_of(o):return int(hashlib.md5(o.encode()).hexdigest()[:8],16)
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt";return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v);return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
val0=set(json.load(open(f"{SCR}/cv_fold0.json"))["val"])
def load(p):
    cg=torch.load(p,map_location="cpu",weights_only=False);gc=AffordanceGNNConfig(**cg["cfg"])
    m=AffordanceGNN(gc);m.load_state_dict(cg["model"]);m.eval();return m,gc
M_base,GC=load("outputs/gnn_synaug8_fold0.pt")
M_rich,_=load("outputs/gnn_synaugrich8_fold0.pt")
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu"))
def enc(w): return vlm.encode_text([w])[0]
def ens(w): return torch.stack([enc(t.format(w=w)) for t in TPL]).mean(0)
emb_exact={v:enc(v) for v in HELD}; emb_hbare={v:enc(HELD[v]) for v in HELD}; emb_hens={v:ens(HELD[v]) for v in HELD}
_c={}
def pred(model,o,e):
    if o not in _c:
        it=ds[o2r[o]];Vn=len(it["vertex_positions"]);r=np.random.default_rng(seed_of(o))
        sel=np.sort(r.choice(Vn,min(SUB,Vn),replace=False));idx=torch.as_tensor(sel,dtype=torch.long)
        knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],GC.knn_k);_c[o]=(it,sel,idx,knn)
    it,sel,idx,knn=_c[o];sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None);z=lambda d:torch.zeros(len(sel),d)
    with torch.no_grad():
        return torch.sigmoid(model(e,vlm_features=z(GC.vlm_dim),dino_vertex=sl("dino_vertex_features"),
            slat_vertex=z(GC.sam3d_dim),vertex_normals=z(GC.normals_dim),vertex_geom=sl("vertex_geom"),knn_idx=knn)).numpy()
print("=== RICH vs BASELINE syn-aug, held-out synonyms (fold0 only) ===")
print(f"{'verb':9s} {'held-out':>11s} {'exact':>6s} {'base-bare':>10s} {'RICH-bare':>10s} {'RICH-ens':>9s}")
EX,BB,RB,RE=[],[],[],[]
for v in HELD:
    ex=[];bb=[];rb=[];re=[]
    for o in sorted(val0):
        if o not in o2r or not clean(o,v):continue
        h=human(o,v)
        if h is None:continue
        p0=pred(M_rich,o,emb_exact[v]);_,sel,_,_=_c[o];y=(h[sel]>=0.5).astype(int)
        if not(0<y.sum()<len(y)):continue
        ex.append(average_precision_score(y,p0))
        bb.append(average_precision_score(y,pred(M_base,o,emb_hbare[v])))
        rb.append(average_precision_score(y,pred(M_rich,o,emb_hbare[v])))
        re.append(average_precision_score(y,pred(M_rich,o,emb_hens[v])))
    if ex:
        EX.append(np.mean(ex));BB.append(np.mean(bb));RB.append(np.mean(rb));RE.append(np.mean(re))
        print(f"{v:9s} {HELD[v]:>11s} {np.mean(ex):6.3f} {np.mean(bb):10.3f} {np.mean(rb):10.3f} {np.mean(re):9.3f}")
if EX:
    print(f"\n{'MACRO':9s} {'':>11s} {np.mean(EX):6.3f} {np.mean(BB):10.3f} {np.mean(RB):10.3f} {np.mean(RE):9.3f}")
    print(f"held-out drop vs exact:  base-bare {np.mean(BB)-np.mean(EX):+.3f}  RICH-bare {np.mean(RB)-np.mean(EX):+.3f}  RICH-ens {np.mean(RE)-np.mean(EX):+.3f}")
    print("(RICH-bare or RICH-ens >> base-bare = richer augmentation helps held-out generalization)")
