"""Radial affordance wheel: one object (neutral grey, center) ringed by verb-conditioned heatmaps.
Same object, same pose in every panel -- only the hot region moves as the verb changes.

Usage:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python render_wheel.py --obj <id> \
      --verbs "grasp,pour,contain,drink from,lift,display" --az 45 --el 25 --out final_figures/task_affordance_wheel.png
"""
from __future__ import annotations
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import numpy as np, torch, trimesh, pyrender
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from wheel_lib import load_model, load_object, encode_verbs, predict

TURBO = cm.get_cmap("turbo")


def look_at(eye, target, up=(0.,1.,0.)):
    eye=np.asarray(eye,float); target=np.asarray(target,float); up=np.asarray(up,float)
    z=eye-target; z/=np.linalg.norm(z); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    M=np.eye(4); M[:3,0]=x; M[:3,1]=y; M[:3,2]=z; M[:3,3]=eye; return M


def _render(mesh, colors, az_deg, el_deg, size=760, fov_deg=42.0):
    v=mesh.vertices.astype(np.float64); center=0.5*(v.max(0)+v.min(0))
    radius=float(np.linalg.norm(v-center,axis=1).max())
    tm=trimesh.Trimesh(mesh.vertices, mesh.faces, process=False); tm.visual.vertex_colors=colors
    pm=pyrender.Mesh.from_trimesh(tm, smooth=True)
    scene=pyrender.Scene(bg_color=[1,1,1,0], ambient_light=[0.75,0.75,0.75]); scene.add(pm)
    az,el=np.radians(az_deg),np.radians(el_deg)
    dist=radius/np.tan(np.radians(fov_deg)/2.)*1.05
    eye=center+dist*np.array([np.cos(el)*np.cos(az),np.sin(el),np.cos(el)*np.sin(az)])
    pose=look_at(eye,center); scene.add(pyrender.PerspectiveCamera(yfov=np.radians(fov_deg)),pose=pose)
    for da in (-35,40):
        la=az+np.radians(da); le=eye+radius*np.array([np.cos(el+0.3)*np.cos(la),np.sin(el+0.5),np.cos(el+0.3)*np.sin(la)])
        scene.add(pyrender.DirectionalLight(color=[1,1,1],intensity=1.6),pose=look_at(le,center))
    r=pyrender.OffscreenRenderer(size,size)
    color,_=r.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES|pyrender.RenderFlags.RGBA); r.delete()
    rgba=color.astype(np.float32)/255.; a=rgba[...,3:4]; rgb=rgba[...,:3]*a+(1.-a)
    return (rgb*255).astype(np.uint8)


def paint_heat(values):
    c=(TURBO(np.clip(values,0,1))[:,:3]*255).astype(np.uint8)
    return np.concatenate([c,np.full((len(c),1),255,np.uint8)],1)


def paint_grey(n, g=205):
    c=np.full((n,3),g,np.uint8)
    return np.concatenate([c,np.full((n,1),255,np.uint8)],1)


def crop(img,pad=14):
    nb=np.any(img<250,axis=2); ys,xs=np.where(nb)
    if len(ys)==0: return img
    y0,y1=max(0,ys.min()-pad),min(img.shape[0],ys.max()+pad)
    x0,x1=max(0,xs.min()-pad),min(img.shape[1],xs.max()+pad)
    return img[y0:y1,x0:x1]


def norm_map(p, lo=0.60, hi=0.999):
    """Contrast-stretch a sigmoid map to [0,1] for vivid, comparable panels."""
    a,b=np.quantile(p,lo),np.quantile(p,hi)
    if b<=a: b=a+1e-6
    return np.clip((p-a)/(b-a),0,1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--obj",required=True)
    ap.add_argument("--verbs",required=True,help="comma-separated verb phrases, in wheel order (clockwise from top)")
    ap.add_argument("--az",type=float,default=45.0); ap.add_argument("--el",type=float,default=25.0)
    ap.add_argument("--title",default="One object, many actions")
    ap.add_argument("--out",default="/home/kraum/Prototype/final_figures/task_affordance_wheel.png")
    a=ap.parse_args()
    verbs=[v.strip() for v in a.verbs.split(",") if v.strip()]

    model,cfg=load_model()
    mesh,pos,kw=load_object(a.obj)
    emb=encode_verbs(verbs)

    grey=crop(_render(mesh, paint_grey(len(mesh.vertices)), a.az, a.el))
    panels={}
    for v in verbs:
        p=predict(model,kw,emb[v])
        panels[v]=crop(_render(mesh, paint_heat(norm_map(p)), a.az, a.el))
        print(f"  rendered {v}  peak={float(np.quantile(p,0.995)):.2f}")

    N=len(verbs); fig=plt.figure(figsize=(12,12)); fig.patch.set_facecolor("white")
    bg=fig.add_axes([0,0,1,1]); bg.set_axis_off(); bg.set_xlim(0,1); bg.set_ylim(0,1)
    cx,cy=0.5,0.5; R=0.345; pw=0.205
    # connectors first (under panels)
    for i in range(N):
        ang=np.radians(90-i*360/N)
        x,y=cx+R*np.cos(ang),cy+R*np.sin(ang)
        bg.add_artist(FancyArrowPatch((cx+0.11*np.cos(ang),cy+0.11*np.sin(ang)),
                                      (x-0.085*np.cos(ang),y-0.085*np.sin(ang)),
                                      arrowstyle='-',color="#c9ccd1",lw=1.4,zorder=1))
    # center object
    cax=fig.add_axes([cx-0.135,cy-0.135,0.27,0.27]); cax.imshow(grey); cax.set_axis_off(); cax.set_zorder(3)
    cax.text(0.5,-0.02,"the geometry",transform=cax.transAxes,ha="center",va="top",fontsize=12,color="#555",style="italic")
    # verb panels
    for i,v in enumerate(verbs):
        ang=np.radians(90-i*360/N); x,y=cx+R*np.cos(ang),cy+R*np.sin(ang)
        ax=fig.add_axes([x-pw/2,y-pw/2,pw,pw]); ax.imshow(panels[v]); ax.set_axis_off(); ax.set_zorder(4)
        lift=1.06 if np.sin(ang)>=0 else -0.06
        ax.text(0.5,lift,v,transform=ax.transAxes,ha="center",
                va="bottom" if np.sin(ang)>=0 else "top",
                fontsize=17,fontweight="bold",color="#1a2b4a")
    fig.text(0.5,0.975,a.title,ha="center",va="top",fontsize=24,fontweight="bold",color="#12233f")
    fig.text(0.5,0.028,"same object · same pose — the verb decides where you act",
             ha="center",va="bottom",fontsize=14,color="#444",style="italic")
    # colorbar
    cbax=fig.add_axes([0.90,0.13,0.015,0.20])
    cb=fig.colorbar(cm.ScalarMappable(cmap=TURBO,norm=matplotlib.colors.Normalize(0,1)),cax=cbax)
    cb.set_ticks([0,1]); cb.set_ticklabels(["low","high"]); cb.set_label("affordance",fontsize=11)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150,facecolor="white"); print("saved",out)


if __name__=="__main__":
    main()
