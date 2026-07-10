"""Final affordance wheel: correct FiLM model (geomclean), sharp poster-style normalization,
minimal text (verb + affordance description ONLY). Center = grey geometry (no label)."""
import os
os.environ.setdefault("PYOPENGL_PLATFORM","egl"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from wheel_lib import load_model, load_object, encode_verbs, predict
from render_wheel import _render, paint_heat, paint_grey, crop

ap=argparse.ArgumentParser()
ap.add_argument("--obj",required=True); ap.add_argument("--spec",required=True)
ap.add_argument("--ckpt",default="/home/kraum/Prototype/outputs/geomclean_fold0/best.pt")
ap.add_argument("--az",type=float,default=20); ap.add_argument("--el",type=float,default=26)
ap.add_argument("--out",required=True); a=ap.parse_args()
pairs=[(s.split("|")[0].strip(),(s.split("|")[1].strip() if "|" in s else "")) for s in a.spec.split(",") if s.strip()]
verbs=[v for v,_ in pairs]; N=len(verbs)
model,cfg=load_model(a.ckpt); mesh,pos,kw=load_object(a.obj,dino_filename=cfg._dino_filename); emb=encode_verbs(verbs)
grey=crop(_render(mesh,paint_grey(len(mesh.vertices)),a.az,a.el,size=820))
panels={}
for v in verbs:
    p=predict(model,kw,emb[v],backbone=cfg._backbone)
    vmax=max(0.05,float(np.quantile(p,0.995)))          # sharp, poster normalization
    panels[v]=crop(_render(mesh,paint_heat(np.clip(p/vmax,0,1)),a.az,a.el,size=820))
    print(f"  {v}: peak={float(np.quantile(p,0.995)):.2f}")

fig=plt.figure(figsize=(12,12)); fig.patch.set_facecolor("white")
bg=fig.add_axes([0,0,1,1]); bg.set_axis_off(); bg.set_xlim(0,1); bg.set_ylim(0,1)
cx,cy=0.5,0.5; R=0.315; pw=0.25
angs={3:[90,210,330],4:[45,135,225,315]}.get(N,[90-i*360/N for i in range(N)])
for ang_d in angs:
    ang=np.radians(ang_d); x,y=cx+R*np.cos(ang),cy+R*np.sin(ang)
    bg.add_artist(FancyArrowPatch((cx+0.12*np.cos(ang),cy+0.12*np.sin(ang)),(x-0.105*np.cos(ang),y-0.105*np.sin(ang)),
                  arrowstyle='-|>',mutation_scale=17,color="#b9bec7",lw=1.7,zorder=1))
cax=fig.add_axes([cx-0.135,cy-0.135,0.27,0.27]); cax.imshow(grey); cax.set_axis_off(); cax.set_zorder(3)
for (v,region),ang_d in zip(pairs,angs):
    ang=np.radians(ang_d); x,y=cx+R*np.cos(ang),cy+R*np.sin(ang)
    ax=fig.add_axes([x-pw/2,y-pw/2,pw,pw]); ax.imshow(panels[v]); ax.set_axis_off(); ax.set_zorder(4)
    up=np.sin(ang)>=0; ly=1.03 if up else -0.03; va="bottom" if up else "top"
    ax.text(0.5,ly,v,transform=ax.transAxes,ha="center",va=va,fontsize=23,fontweight="bold",color="#16233c")
    if region:
        ax.text(0.5,ly+(0.085 if up else -0.085),f"→ the {region}",transform=ax.transAxes,ha="center",va=va,
                fontsize=14,color="#c0392b",style="italic")
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=150,facecolor="white"); print("saved",out)
