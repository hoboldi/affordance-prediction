"""CHECK 3: decompose the 0.52. Does the pretrained-only head (distilled on GEAL pseudolabels, NO human
finetune) beat GEAL uniformly across verbs, or is the 0.46->0.52 gain driven by one/two verbs?
Feeds the features the pretrain was trained on. Held-out clean human GT. CPU."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
DR = "/home/datasets/customDatasets/cmr2/reconstructions"
V = ["contain","pour","sit","move","display","grasp","press","lift"]
GEO = {"contain","pour","sit","move","lift"}; APP = {"grasp","display","press"}
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def geal(o,v):
    p=f"{DR}/{o}/vertex_pseudolabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v); return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
valset=set()
for k in range(5): valset|=set(json.load(open(f"{SCR}/cv_fold{k}.json"))["val"])
cp=torch.load("outputs/gnn_pretrain_k24_8v.pt",map_location="cpu",weights_only=False)
cfg=AffordanceGNNConfig(**cp["cfg"]); m=AffordanceGNN(cfg); m.load_state_dict(cp["model"]); m.eval()
print(f"pretrain cfg dims: vlm={cfg.vlm_dim} dino={cfg.dino_vertex_dim} slat={cfg.sam3d_dim} norm={cfg.normals_dim} geom={cfg.geom_dim}")
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu")); emb={v:vlm.encode_text([v])[0] for v in V}
byv={v:{"g":[],"p":[]} for v in V}
for o in sorted(valset):
    if o not in o2r: continue
    it=ds[o2r[o]]; Vn=len(it["vertex_positions"]); r=np.random.default_rng(seed_of(o))
    sel=np.sort(r.choice(Vn,min(24000,Vn),replace=False)); idx=torch.as_tensor(sel,dtype=torch.long)
    knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],cfg.knn_k)
    sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None); z=lambda d:torch.zeros(len(sel),d)
    for v in V:
        g=geal(o,v); h=human(o,v)
        if g is None or h is None or not clean(o,v): continue
        y=(h[sel]>=0.5).astype(int)
        if not (0<y.sum()<len(y)): continue
        with torch.no_grad():
            p=torch.sigmoid(m(emb[v],vlm_features=sl("vertex_features") if sl("vertex_features") is not None else z(cfg.vlm_dim),
                dino_vertex=sl("dino_vertex_features"),slat_vertex=sl("slat_vertex_features") if sl("slat_vertex_features") is not None else z(cfg.sam3d_dim),
                vertex_normals=sl("vertex_normals") if sl("vertex_normals") is not None else z(cfg.normals_dim),
                vertex_geom=sl("vertex_geom"),knn_idx=knn)).numpy().reshape(-1)
        byv[v]["g"].append(average_precision_score(y,g[sel])); byv[v]["p"].append(average_precision_score(y,p))
print(f"\n=== Pretrained-only head vs GEAL, per verb (held-out clean) ===")
print(f"{'verb':9s} {'n':>3s} {'GEAL':>6s} {'Pretr':>6s} {'Δ':>7s} {'beats?':>7s}")
gm,pm=[],[]; wins=0
for v in V:
    if byv[v]["p"]:
        gg,pp=np.mean(byv[v]["g"]),np.mean(byv[v]["p"]); gm.append(gg); pm.append(pp)
        w = pp>gg; wins+=w
        print(f"{v:9s} {len(byv[v]['p']):3d} {gg:6.3f} {pp:6.3f} {pp-gg:+7.3f} {'YES' if w else 'no':>7s}")
print(f"{'MACRO':9s}     {np.mean(gm):6.3f} {np.mean(pm):6.3f} {np.mean(pm)-np.mean(gm):+7.3f}   verbs beaten: {wins}/{len(pm)}")
print("uniform wins => clean domain-adaptation story (claim it). driven by 1-2 verbs => drop the 0.46->0.52 claim.")
