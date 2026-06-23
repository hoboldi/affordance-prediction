"""Validate the feature-render overhaul: OLD (single 6-ring, equal fusion, lit) vs NEW (multi-ring,
facing-weighted, unlit, auto-fit) per-vertex DINO features as PCA->RGB on a mesh, + a view montage.
Looking for: the 6-fold azimuthal star to vanish (smooth field), tighter framing, true colour."""
from __future__ import annotations
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, trimesh
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.mesh_loading import load_mesh
from rendering.renderer import render_mesh_views
from projection.project_to_mesh import ViewProjectionInputs, project_views_to_vertices, vertex_pca_colors
from utils.config import load_config
from generate_vertex_dino import DinoExtractor

import glob
cands = sorted(glob.glob("/home/datasets/customDatasets/cmr2/reconstructions/bowl__*"))
D = cands[0] if cands else sorted(glob.glob("/home/datasets/customDatasets/cmr2/reconstructions/*"))[0]
print("object:", os.path.basename(D))
mesh = load_mesh(D + "/mesh.glb", process=False)
vnormals = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).vertex_normals
ext = DinoExtractor("cuda")

def run(cfg, weighted):
    views = render_mesh_views(mesh, cfg)
    patches = ext.patches([v.rgb for v in views])
    pv = [ViewProjectionInputs(vertex_uv=v.vertex_uv, vertex_visible=v.vertex_visible, patches=p,
            render_height=v.rgb.shape[0], render_width=v.rgb.shape[1], camera_position=v.camera_pose[:3, 3])
          for v, p in zip(views, patches)]
    kw = dict(vertex_normals=vnormals, vertex_positions=mesh.vertices) if weighted else {}
    vs = project_views_to_vertices(pv, **kw)
    return vs.features, views

import copy
cfg_old = copy.deepcopy(load_config()); r = cfg_old["rendering"]; r["backend"] = "mesh"
r["elevation_rings_deg"] = None; r["auto_fit_framing"] = False; r["unlit_albedo"] = False; r["num_views"] = 6
f_old, views_old = run(cfg_old, weighted=False)

cfg_new = copy.deepcopy(load_config()); cfg_new["rendering"]["backend"] = "mesh"
f_new, views_new = run(cfg_new, weighted=True)
print("OLD views", len(views_old), "NEW views", len(views_new))

c_old, c_new = vertex_pca_colors(f_old), vertex_pca_colors(f_new)
xyz = mesh.vertices
rng = np.random.default_rng(0); sel = rng.choice(len(xyz), min(15000, len(xyz)), replace=False)

fig = plt.figure(figsize=(16, 8))
# top row: 4 NEW rendered views (framing / texture / unlit / elevation)
for i, vi in enumerate(np.linspace(0, len(views_new) - 1, 4).astype(int)):
    ax = fig.add_subplot(2, 4, i + 1); ax.imshow(views_new[vi].rgb); ax.set_axis_off()
    ax.set_title(f"NEW view {vi}", fontsize=9)
# bottom row: OLD vs NEW PCA->RGB on mesh, top-down-ish (star is clearest looking into the bowl)
for j, (name, col) in enumerate([("OLD feat PCA (a)", c_old), ("OLD (b)", c_old), ("NEW feat PCA (a)", c_new), ("NEW (b)", c_new)]):
    ax = fig.add_subplot(2, 4, 5 + j, projection="3d")
    ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=col[sel] / 255.0, s=3)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[2]-h, mid[2]+h); ax.set_zlim(mid[1]-h, mid[1]+h)
    ax.view_init(elev=(75 if "(a)" in name else 30), azim=-60); ax.set_axis_off(); ax.set_title(name, fontsize=9)
fig.suptitle(f"Feature-render fix on {os.path.basename(D)} — top: new views; bottom: 6-star check (OLD vs NEW DINO PCA)", fontsize=11)
out = "outputs/renders/diag_render_fix.png"
os.makedirs("outputs/renders", exist_ok=True); fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
