"""CHECK A: is the appearance DINO-reliance INHERITED from GEAL, or LEARNED in human finetuning?
Pretrained-only head (distilled, NO human FT), full features. Per class: absolute AUPRC AND DINO-drop
(feed all vs zero DINO). Compare to finetuned lean8 (App 0.851/drop 0.245, Geo 0.870/drop 0.06).
If pretrained App is near-chance and/or its DINO-drop is small, the reliance was learned from human labels. CPU."""
import os, sys, json, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "experiments/cv_harness"
V = ["contain","pour","sit","move","display","grasp","press","lift"]
GEO = {"contain","pour","sit","move","lift"}; APP = {"grasp","display","press"}
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v); return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005
valset=set()
for k in range(5): valset|=set(json.load(open(f"{SCR}/cv_fold{k}.json"))["val"])
cp=torch.load("outputs/gnn_pretrain_k24_8v.pt",map_location="cpu",weights_only=False)
cfg=AffordanceGNNConfig(**cp["cfg"]); m=AffordanceGNN(cfg); m.load_state_dict(cp["model"]); m.eval()
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu")); emb={v:vlm.encode_text([v])[0] for v in V}
byv={v:{"full":[],"nodino":[]} for v in V}
for o in sorted(valset):
    if o not in o2r: continue
    it=ds[o2r[o]]; Vn=len(it["vertex_positions"]); r=np.random.default_rng(seed_of(o))
    sel=np.sort(r.choice(Vn,min(24000,Vn),replace=False)); idx=torch.as_tensor(sel,dtype=torch.long)
    knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],cfg.knn_k)
    sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None); z=lambda d:torch.zeros(len(sel),d)
    for v in V:
        h=human(o,v)
        if h is None or not clean(o,v): continue
        y=(h[sel]>=0.5).astype(int)
        if not (0<y.sum()<len(y)): continue
        args=dict(vlm_features=sl("vertex_features") if sl("vertex_features") is not None else z(cfg.vlm_dim),
            slat_vertex=sl("slat_vertex_features") if sl("slat_vertex_features") is not None else z(cfg.sam3d_dim),
            vertex_normals=sl("vertex_normals") if sl("vertex_normals") is not None else z(cfg.normals_dim),
            vertex_geom=sl("vertex_geom"),knn_idx=knn)
        with torch.no_grad():
            pf=torch.sigmoid(m(emb[v],dino_vertex=sl("dino_vertex_features"),**args)).numpy().reshape(-1)
            pn=torch.sigmoid(m(emb[v],dino_vertex=z(cfg.dino_vertex_dim),**args)).numpy().reshape(-1)
        byv[v]["full"].append(average_precision_score(y,pf)); byv[v]["nodino"].append(average_precision_score(y,pn))
def cls_mean(d,vs): return np.mean([np.mean(byv[v][d]) for v in vs if byv[v][d]])
print("=== CHECK A: pretrained-only head (distilled, NO human FT), full features ===")
print(f"{'verb':9s} {'full':>6s} {'noDINO':>7s} {'DINO-drop':>10s}")
for v in V:
    if byv[v]["full"]:
        f=np.mean(byv[v]["full"]); n=np.mean(byv[v]["nodino"]); print(f"{v:9s} {f:6.3f} {n:7.3f} {f-n:+10.3f}")
gf,af=cls_mean("full",GEO),cls_mean("full",APP); gn,an=cls_mean("nodino",GEO),cls_mean("nodino",APP)
print(f"\n{'class':11s} {'pretr abs':>10s} {'pretr DINO-drop':>16s} {'| finetuned abs':>16s} {'FT DINO-drop':>13s}")
print(f"{'Geometric':11s} {gf:10.3f} {gf-gn:16.3f} {'| 0.870':>16s} {'0.06':>13s}")
print(f"{'Appearance':11s} {af:10.3f} {af-an:16.3f} {'| 0.851':>16s} {'0.245':>13s}")
print("\nRead: if pretrained Appearance DINO-drop << finetuned 0.245 (and/or pretr App abs is low), the")
print("appearance DINO-reliance was LEARNED in human finetuning, not inherited from GEAL. If pretr App")
print("DINO-drop is already large, it was inherited.")
