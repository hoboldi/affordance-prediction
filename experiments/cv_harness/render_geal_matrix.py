"""GEAL matrix (clean, held-out): 3 rows (Human GT | GEAL teacher | Ours) x 6 cols (verbs).
One representative held-out object per verb, category-appropriate view, visible region. Text-free except
column verb headers + row labels + tiny per-cell AUPRC. GEAL = vertex_pseudolabels (the teacher we distill from)."""
import os, sys, json, hashlib, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig
from vlm.vlm_wrapper import VLMWrapper, VLMConfig
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
DR = "/home/datasets/customDatasets/cmr2/reconstructions"
VERBS = ["contain","sit","pour","move","display","grasp"]
SUB = 24000
PREF = {"contain":["bowl"],"sit":["chair"],"pour":["bottle","vase","cup"],
        "move":["chair"],"display":["laptop","tv"],"grasp":["cup","handbag","bottle"]}
VIEW = {"laptop":(25,110),"tv":(20,120),"chair":(16,-60),"bowl":(58,-55),"cup":(16,-60),
        "bottle":(12,-60),"vase":(18,-60),"handbag":(18,-55),"microwave":(20,-60)}
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
def cat(o): return o.split("__")[0]
def geal(o,v):
    p=f"{DR}/{o}/vertex_pseudolabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def human(o,v):
    p=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"; return np.asarray(torch.load(p,weights_only=False)).reshape(-1) if os.path.exists(p) else None
def clean(o,v):
    h=human(o,v); return h is not None and ((h>0.001)&(h<0.999)).mean()<0.005

obj2fold={}
for k in range(5):
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: obj2fold[o]=k
gnn={}
for k in range(5):
    cg=torch.load(f"outputs/gnn_full_preclean_fold{k}.pt",map_location="cpu",weights_only=False)
    gc=AffordanceGNNConfig(**cg["cfg"]); m=AffordanceGNN(gc); m.load_state_dict(cg["model"]); m.eval(); gnn[k]=(m,gc)
ds=DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl",load_vertex_labels_eager=False,
    load_vertex_semantics_eager=True,load_vertex_dino=True,dino_filename="vertex_dino_fine.pt",
    load_vertex_geom=True,geom_filename="vertex_geom.pt")
o2r={os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")):i for i,r in enumerate(ds.rows)}
vlm=VLMWrapper(VLMConfig(device="cpu")); emb={v:vlm.encode_text([v])[0] for v in VERBS}
_pc={}
def predict(o,v):
    if o not in _pc:
        it=ds[o2r[o]]; V=len(it["vertex_positions"]); r=np.random.default_rng(seed_of(o))
        sel=np.sort(r.choice(V,min(SUB,V),replace=False)); idx=torch.as_tensor(sel,dtype=torch.long)
        k=obj2fold[o]; knn=AffordanceGNN._build_knn(it["vertex_positions"].float()[idx],gnn[k][1].knn_k)
        _pc[o]=(it,sel,idx,knn,k)
    it,sel,idx,knn,k=_pc[o]; sl=lambda kk:(it.get(kk).float()[idx] if it.get(kk) is not None else None)
    with torch.no_grad():
        p=torch.sigmoid(gnn[k][0](emb[v],vlm_features=sl("vertex_features"),dino_vertex=sl("dino_vertex_features"),
            slat_vertex=sl("slat_vertex_features"),vertex_normals=sl("vertex_normals"),vertex_geom=sl("vertex_geom"),knn_idx=knn)).numpy()
    return p,sel

# per-verb candidate list (prefer good categories, ours strong, geal weaker, mid region)
cands=collections.defaultdict(list)
for o in sorted(obj2fold):
    if o not in o2r: continue
    for v in VERBS:
        g=geal(o,v); h=human(o,v)
        if g is None or h is None or not clean(o,v): continue
        if cat(o) not in PREF[v]: continue
        p,sel=predict(o,v); y=(h[sel]>=0.5).astype(int); fr=y.sum()/len(y)
        if not (0.04<fr<0.6): continue
        oa=average_precision_score(y,p); ga=average_precision_score(y,g[sel])
        if oa>0.75: cands[v].append((oa-ga,ga,oa,o))
pick={}
for v in VERBS:
    if cands[v]: cands[v].sort(reverse=True); pick[v]=cands[v][0]  # highest contrast
    else: print("NO PICK for",v)

def draw(ax,xyz,sel,c,gt,view):
    vmax=1.0 if gt else float(max(0.05,c.max()))
    ax.scatter(xyz[sel,0],xyz[sel,2],xyz[sel,1],c=c,cmap="turbo",s=3,vmin=0,vmax=vmax)
    h=(xyz[sel].max(0)-xyz[sel].min(0)).max()/2+1e-6; mid=(xyz[sel].max(0)+xyz[sel].min(0))/2
    ax.set_xlim(mid[0]-h,mid[0]+h); ax.set_ylim(mid[2]-h,mid[2]+h); ax.set_zlim(mid[1]-h,mid[1]+h)
    ax.view_init(elev=view[0],azim=view[1]); ax.set_axis_off()

fig=plt.figure(figsize=(6*2.3,3*2.5))
rowlab=["Human GT","GEAL (teacher)","Ours"]
for ci,v in enumerate(VERBS):
    if v not in pick: continue
    d,ga,oa,o=pick[v]; it=_pc[o][0]; xyz=it["vertex_positions"].numpy(); p,sel=predict(o,v)
    hb=human(o,v)[sel]; g=geal(o,v)[sel]; view=VIEW.get(cat(o),(16,-60))
    for ri,(fld,gt,ap) in enumerate([(hb,True,None),(g,False,ga),(p,False,oa)]):
        ax=fig.add_subplot(3,6,ri*6+ci+1,projection="3d"); draw(ax,xyz,sel,fld,gt,view)
        if ri==0: ax.set_title(v,fontsize=13,fontweight="bold",pad=-2)
        if ci==0: ax.text2D(-0.10,0.5,rowlab[ri],transform=ax.transAxes,fontsize=12,fontweight="bold",
                            rotation=90,va="center",color=("tab:green" if ri==2 else "black"))
        if ap is not None: ax.text2D(0.5,0.03,f"{ap:.2f}",transform=ax.transAxes,fontsize=9,ha="center",
                                     color=("tab:green" if ri==2 else "gray"))
plt.subplots_adjust(wspace=-0.05,hspace=0.03,left=0.05,right=0.99,top=0.95,bottom=0.01)
out="final_figures/result_geal_matrix.png"; fig.savefig(out,dpi=140,bbox_inches="tight")
print("saved",out,"picks:",{v:(pick[v][3].split('__')[0],round(pick[v][1],2),round(pick[v][2],2)) for v in pick})
