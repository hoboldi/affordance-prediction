"""Pretty base-geometry render: PBR porcelain material + 3-point lighting + 2x supersampling,
transparent background. Overwrites base_geometry.png."""
import os
os.environ.setdefault("PYOPENGL_PLATFORM","egl"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import numpy as np, trimesh, pyrender
from PIL import Image
from wheel_lib import load_object
from render_wheel import look_at

def render_pretty(mesh, az_deg=20, el_deg=26, size=2000, fov_deg=40.0,
                  base=(0.86,0.85,0.87), rough=0.48):
    v=mesh.vertices.astype(np.float64); center=0.5*(v.max(0)+v.min(0))
    radius=float(np.linalg.norm(v-center,axis=1).max())
    tm=trimesh.Trimesh(mesh.vertices,mesh.faces,process=False)
    mat=pyrender.MetallicRoughnessMaterial(baseColorFactor=[*base,1.0],metallicFactor=0.0,
                                           roughnessFactor=rough,smooth=True,
                                           emissiveFactor=[0.03,0.03,0.035])
    pm=pyrender.Mesh.from_trimesh(tm,material=mat,smooth=True)
    scene=pyrender.Scene(bg_color=[1,1,1,0],ambient_light=[0.22,0.22,0.25]); scene.add(pm)
    az,el=np.radians(az_deg),np.radians(el_deg)
    dist=radius/np.tan(np.radians(fov_deg)/2.)*1.08
    eye=center+dist*np.array([np.cos(el)*np.cos(az),np.sin(el),np.cos(el)*np.sin(az)])
    scene.add(pyrender.PerspectiveCamera(yfov=np.radians(fov_deg)),pose=look_at(eye,center))
    # 3-point lighting, positioned relative to camera azimuth
    def light_pose(daz_deg, del_deg):
        la=az+np.radians(daz_deg); le=el+np.radians(del_deg)
        p=center+radius*np.array([np.cos(le)*np.cos(la),np.sin(le),np.cos(le)*np.sin(la)])
        return look_at(p,center)
    scene.add(pyrender.DirectionalLight(color=[1.0,0.98,0.95],intensity=4.2), pose=light_pose(-42, 32))  # key (warm, upper-left)
    scene.add(pyrender.DirectionalLight(color=[0.9,0.93,1.0],intensity=1.4),  pose=light_pose(55, 5))    # fill (cool, right)
    scene.add(pyrender.DirectionalLight(color=[1,1,1],intensity=3.0),          pose=light_pose(160, 48))  # rim (behind-top)
    r=pyrender.OffscreenRenderer(size,size)
    try:
        color,_=r.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES|pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    except Exception:
        color,_=r.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES|pyrender.RenderFlags.RGBA)
    r.delete(); return color

def crop_alpha(img,pad=16):
    a=img[...,3]>8; ys,xs=np.where(a)
    y0,y1=max(0,ys.min()-pad),min(img.shape[0],ys.max()+pad); x0,x1=max(0,xs.min()-pad),min(img.shape[1],xs.max()+pad)
    return img[y0:y1,x0:x1]

ap=argparse.ArgumentParser()
ap.add_argument("--obj",default="cup__12_100_593__v000")
ap.add_argument("--az",type=float,default=20); ap.add_argument("--el",type=float,default=26)
ap.add_argument("--out",default="final_figures/task_affordance_wheel_parts/base_geometry.png"); a=ap.parse_args()
mesh,pos,kw=load_object(a.obj,dino_filename="vertex_dino_fine.pt")
img=crop_alpha(render_pretty(mesh,a.az,a.el))
im=Image.fromarray(img); im=im.resize((im.width//2, im.height//2), Image.LANCZOS)   # supersample down
im.save(a.out); print("saved",a.out, im.size)
