"""3b middle number: eval the PRETRAINED head (gnn_pretrain_k24, distilled on GEAL pseudolabels, NO human FT)
on human GT — same object set as the GEAL scan — verb-balanced (macro). Gives GEAL < pretrained < finetuned."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR="/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
DR="/home/datasets/customDatasets/cmr2/reconstructions"
VERBS=["contain","sit","pour","move","display","grasp"]; GEO=["contain","pour","sit","move"]; APP=["grasp","display"]
SUB=24000; dev=torch.device("cuda")
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8],16)
def geal(o,v):
    p=f"{DR}/{o}/vertex_pseudolabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v); return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
cp=torch.load("outputs/gnn_pretrain_k24.pt",map_location=dev,weights_only=False)
cfg=AffordanceGNNConfig(**cp["cfg"]); m=AffordanceGNN(cfg).to(dev); m.load_state_dict(cp["model"]); m.eval()
print("pretrained cfg dims: vlm=%d dino=%d slat=%d norm=%d geom=%d"%(cfg.vlm_dim,cfg.dino_vertex_dim,cfg.sam3d_dim,cfg.normals_dim,cfg.geom_dim))
valset=set()
for k in range(5): valset|=set(json.load(open(f"{SCR}/cv_fold{k}.json"))["val"])
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu")); emb={v:vlm.encode_text([v])[0].to(dev) for v in VERBS}
byv={v:{"pre":[],"geal":[]} for v in VERBS}
for o in sorted(valset):
    if o not in o2r: continue
    it=ds[o2r[o]]; V=len(it["vertex_positions"]); r=np.random.default_rng(seed_of(o))
    sel=np.sort(r.choice(V,min(SUB,V),replace=False)); idx=torch.as_tensor(sel,dtype=torch.long)
    knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],cfg.knn_k).to(dev)
    sl=lambda kk:(it.get(kk).float()[idx].to(dev) if it.get(kk) is not None else None)
    for v in VERBS:
        g=geal(o,v); h=human(o,v)
        if g is None or h is None or not clean(o,v): continue
        y=(h[sel]>=0.5).astype(int)
        if not (0<y.sum()<len(y)): continue
        with torch.no_grad():
            p=torch.sigmoid(m(emb[v],vlm_features=sl("vertex_features"),dino_vertex=sl("dino_vertex_features"),
                slat_vertex=sl("slat_vertex_features"),vertex_normals=sl("vertex_normals"),vertex_geom=sl("vertex_geom"),knn_idx=knn)).cpu().numpy()
        byv[v]["pre"].append(average_precision_score(y,p)); byv[v]["geal"].append(average_precision_score(y,g[sel]))
print(f"{'verb':9s} {'n':>3s} {'GEAL':>6s} {'PRETRAINED':>11s}")
pm,gm=[],[]
for v in VERBS:
    if byv[v]["pre"]:
        pv_=np.mean(byv[v]["pre"]); gv=np.mean(byv[v]["geal"]); pm.append(pv_); gm.append(gv)
        print(f"{v:9s} {len(byv[v]['pre']):3d} {gv:6.3f} {pv_:11.3f}")
def grp(d,vs): return np.mean([np.mean(byv[v][d]) for v in vs if byv[v][d]])
print(f"\nMACRO (verb-balanced):  GEAL={np.mean(gm):.3f}  PRETRAINED={np.mean(pm):.3f}")
print(f"  by class PRETRAINED:  GEO={grp('pre',GEO):.3f}  APP={grp('pre',APP):.3f}")
