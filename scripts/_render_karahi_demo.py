"""Render karahi human-GT vs GNN-prediction affordance maps (contain + grasp) as painted meshes.

Matched camera/style for a clean GT-vs-prediction comparison. Outputs into final_figures/.
"""
from __future__ import annotations
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys
from pathlib import Path
import numpy as np
import trimesh
import pyrender
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "cam_karahi_feat" / "karahi__demo__v000"
OUT = ROOT / "final_figures"
TURBO = cm.get_cmap("turbo")


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4); M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = eye
    return M


def render_painted(mesh: trimesh.Trimesh, values: np.ndarray, az_deg=45.0, el_deg=32.0,
                   size=900, fov_deg=42.0) -> np.ndarray:
    v = mesh.vertices.astype(np.float64)
    center = 0.5 * (v.max(0) + v.min(0))
    radius = float(np.linalg.norm(v - center, axis=1).max())
    colors = (TURBO(np.clip(values, 0, 1))[:, :3] * 255).astype(np.uint8)
    colors = np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], 1)
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    tm.visual.vertex_colors = colors
    pm = pyrender.Mesh.from_trimesh(tm, smooth=True)
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.72, 0.72, 0.72])
    scene.add(pm)
    az, el = np.radians(az_deg), np.radians(el_deg)
    dist = radius / np.tan(np.radians(fov_deg) / 2.0) * 1.05
    eye = center + dist * np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    pose = look_at(eye, center)
    cam = pyrender.PerspectiveCamera(yfov=np.radians(fov_deg))
    scene.add(cam, pose=pose)
    # two soft directional lights from camera-ish angles (gentle, to keep colormap faithful)
    for da in (-35, 40):
        la = az + np.radians(da)
        le = eye + radius * np.array([np.cos(el + 0.3) * np.cos(la), np.sin(el + 0.5), np.cos(el + 0.3) * np.sin(la)])
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.5), pose=look_at(le, center))
    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.RGBA)
    r.delete()
    # composite RGBA over white
    rgba = color.astype(np.float32) / 255.0
    a = rgba[..., 3:4]
    rgb = rgba[..., :3] * a + (1.0 - a)  # white background
    return (rgb * 255).astype(np.uint8)


def crop_bg(img: np.ndarray, pad=18) -> np.ndarray:
    nonbg = np.any(img < 250, axis=2)
    ys, xs = np.where(nonbg)
    if len(ys) == 0:
        return img
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1]


def main():
    mesh = trimesh.load(FEAT / "mesh.glb", process=False, force="mesh")
    preds = np.load(FEAT.parent / "preds.npz")
    verbs = ["contain", "grasp"]
    tiles = {}
    for verb in verbs:
        gt = np.load(ROOT / "human_gt_labels" / f"karahi__demo__v000__{verb}.pt", allow_pickle=True) if False else None
        import torch
        gt = torch.load(ROOT / "human_gt_labels" / f"karahi__demo__v000__{verb}.pt",
                        map_location="cpu", weights_only=False)["labels"].numpy().astype(float)
        pred = preds[verb]
        tiles[(verb, "gt")] = crop_bg(render_painted(mesh, gt))
        tiles[(verb, "pred")] = crop_bg(render_painted(mesh, pred))
        # individual PNGs
        for kind in ("gt", "pred"):
            plt.imsave(OUT / f"karahi_{verb}_{kind}.png", tiles[(verb, kind)])
        print(f"{verb}: rendered GT + pred")

    # combined 2x2 figure: rows=verb, cols=GT|Prediction
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.6))
    col_titles = ["Human ground truth", "Model prediction"]
    for ci, kind in enumerate(("gt", "pred")):
        for ri, verb in enumerate(verbs):
            ax = axes[ri, ci]
            ax.imshow(tiles[(verb, kind)]); ax.set_axis_off()
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=15, fontweight="bold", pad=8)
            if ci == 0:
                ax.text(-0.06, 0.5, verb, transform=ax.transAxes, fontsize=15, fontweight="bold",
                        rotation=90, va="center", ha="center")
    # shared turbo colorbar
    sm = cm.ScalarMappable(cmap=TURBO, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, location="right")
    cb.set_label("affordance score", fontsize=12)
    fig.suptitle("Karahi — human GT vs GNN prediction (verb-conditioned)", fontsize=16, y=0.98)
    fig.savefig(OUT / "karahi_gt_vs_pred.png", dpi=140, bbox_inches="tight")
    print("saved", OUT / "karahi_gt_vs_pred.png")


if __name__ == "__main__":
    main()
