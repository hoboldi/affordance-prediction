"""Final planned step: does averaging the +geom model with the plain full-FT model beat +geom alone?
fold0 val, 6 trained verbs, human GT. CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain","sit","pour","move","display","grasp"]
SUB=25000; rng=np.random.default_rng(0); f=lambda t: t.float() if t is not None else None
def load(p):
    c=torch.load(p,map_location="cpu",weights_only=False); m=AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval(); return m
base=load("outputs/basefull_fold0/last.pt"); geom=load("outputs/geomcv_fold0/last.pt")
bcfg=mlp_head_config_from_model_cfg(torch.load("outputs/geomcv_fold0/last.pt",map_location="cpu",weights_only=False)["model_cfg"])
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                   load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename,
                   load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu")); emb={v:vlm.encode_text([v])[0] for v in TRAINED}
val=json.load(open(f"{SCR}/cv_fold0.json"))["val"]
byv=collections.defaultdict(lambda:{"base":[],"geom":[],"ens":[]})
for o in val:
    if o not in o2r: continue
    it=None
    for v in TRAINED:
        hf=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb=(np.asarray(torch.load(hf,weights_only=False)).reshape(-1)>=0.5).astype(int)
        if not (0<hb.sum()<len(hb)): continue
        if it is None: it=ds[o2r[o]]
        V=len(hb); sel=rng.choice(V,min(SUB,V),replace=False); idx=torch.as_tensor(sel,dtype=torch.long)
        sl=lambda k:(it.get(k).float()[idx] if it.get(k) is not None else None)
        kw=dict(vlm_features=sl("vertex_features"),dino_cls=f(it.get("dino_cls")),ss_dino_cls=f(it.get("ss_dino_cls")),
                vertex_normals=sl("vertex_normals"),vertex_positions=None,dino_vertex=sl("dino_vertex_features"),
                slat_vertex=sl("slat_vertex_features"))
        with torch.no_grad():
            pb=torch.sigmoid(base(emb[v],**kw)).numpy()
            pg=torch.sigmoid(geom(emb[v],vertex_geom=sl("vertex_geom"),**kw)).numpy()
        byv[v]["base"].append(average_precision_score(hb[sel],pb))
        byv[v]["geom"].append(average_precision_score(hb[sel],pg))
        byv[v]["ens"].append(average_precision_score(hb[sel],(pb+pg)/2))
print(f"=== fold0 val: base vs +geom vs ensemble(avg) ===")
print(f"{'verb':9s} {'base':>7s} {'+geom':>7s} {'ens':>7s}")
mb,mg,me=[],[],[]
for v in TRAINED:
    b,g,e=np.mean(byv[v]['base']),np.mean(byv[v]['geom']),np.mean(byv[v]['ens'])
    mb.append(b); mg.append(g); me.append(e)
    print(f"{v:9s} {b:7.3f} {g:7.3f} {e:7.3f}")
print(f"{'MEAN':9s} {np.mean(mb):7.3f} {np.mean(mg):7.3f} {np.mean(me):7.3f}")
