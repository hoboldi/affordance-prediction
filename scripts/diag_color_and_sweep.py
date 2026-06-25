"""(1) Does unlit_albedo preserve real color? Compare mesh vertex-colour vs rendered-pixel saturation
on a COLOURED object (bottle) + a white object (bowl). (2) View-count sweep (12/16/24) — smallest that
keeps the 6-fold star gone."""
from __future__ import annotations
import os, sys, copy, glob
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.mesh_loading import load_mesh
from rendering.renderer import render_mesh_views
from projection.project_to_mesh import ViewProjectionInputs, project_views_to_vertices, vertex_pca_colors
from utils.config import load_config
from generate_vertex_dino import DinoExtractor

def first(pat):
    g = sorted(glob.glob(f"/home/datasets/customDatasets/cmr2/reconstructions/{pat}")); return g[0] if g else None
BOTTLE = first("bottle__*"); BOWL = first("bowl__*")
ext = DinoExtractor("cuda")

def sat(rgb):  # mean per-pixel (max-min) over non-bg
    obj = rgb[rgb.sum(-1) > 12].astype(np.float32); return float((obj.max(1) - obj.min(1)).mean()), obj.mean(0).round(1)

def render_new(D):
    mesh = load_mesh(D + "/mesh.glb", process=False)
    cfg = copy.deepcopy(load_config()); cfg["rendering"]["backend"] = "mesh"
    return mesh, render_mesh_views(mesh, cfg)

print("=== COLOR FAITHFULNESS (unlit albedo) ===")
imgs = {}
for name, D in [("bottle", BOTTLE), ("bowl", BOWL)]:
    mesh, views = render_new(D)
    vc = np.asarray(mesh.vertex_colors)[:, :3].astype(np.float32)
    mesh_sat = float((vc.max(1) - vc.min(1)).mean())
    r_sat, r_mean = sat(views[0].rgb)
    print(f"{name}: mesh vtx-colour sat={mesh_sat:.1f} (mean {vc.mean(0).round(1)}) | render sat={r_sat:.1f} (mean {r_mean})")
    imgs[name] = views[0].rgb

def sweep_feats(D, rings, az):
    mesh = load_mesh(D + "/mesh.glb", process=False)
    cfg = copy.deepcopy(load_config()); r = cfg["rendering"]; r["backend"] = "mesh"
    r["elevation_rings_deg"] = rings; r["azimuths_per_ring"] = az
    views = render_mesh_views(mesh, cfg)
    vn = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).vertex_normals
    pv = [ViewProjectionInputs(vertex_uv=v.vertex_uv, vertex_visible=v.vertex_visible, patches=p,
            render_height=v.rgb.shape[0], render_width=v.rgb.shape[1], camera_position=v.camera_pose[:3, 3])
          for v, p in zip(views, ext.patches([v.rgb for v in views]))]
    vs = project_views_to_vertices(pv, vertex_normals=vn, vertex_positions=mesh.vertices)
    return mesh.vertices, vertex_pca_colors(vs.features), len(views)

CONF = [("12 (2x6)", [33.0, 52.0], 6), ("16 (2x8)", [33.0, 52.0], 8), ("24 (3x8)", [27.0, 43.0, 58.0], 8)]
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(2, 3, 1); ax.imshow(imgs["bottle"]); ax.set_axis_off(); ax.set_title("bottle unlit (colour check)", fontsize=10)
ax = fig.add_subplot(2, 3, 2); ax.imshow(imgs["bowl"]); ax.set_axis_off(); ax.set_title("bowl unlit (white object)", fontsize=10)
rng = np.random.default_rng(0)
for i, (label, rings, az) in enumerate(CONF):
    xyz, col, nv = sweep_feats(BOWL, rings, az)
    sel = rng.choice(len(xyz), min(15000, len(xyz)), replace=False)
    a = fig.add_subplot(2, 3, 4 + i, projection="3d")
    a.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=col[sel] / 255.0, s=3)
    h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
    a.set_xlim(mid[0]-h, mid[0]+h); a.set_ylim(mid[2]-h, mid[2]+h); a.set_zlim(mid[1]-h, mid[1]+h)
    a.view_init(elev=75, azim=-60); a.set_axis_off(); a.set_title(f"bowl PCA top-down — {label} views", fontsize=10)
    print(f"sweep {label}: {nv} views")
fig.suptitle("Top: unlit colour faithfulness | Bottom: view-count sweep (6-star check)", fontsize=11)
out = "outputs/renders/diag_color_and_sweep.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out)
