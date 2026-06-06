"""
Transfer AffordSplat affordance annotations onto SAM3D mesh vertices.

The annotation file (``GS_anno_<id>.ply``) holds only the Gaussians of one affordance
region, in the **original AffordSplat frame**. The SAM3D ``mesh.glb`` is in SAM3D's own
reconstructed pose (different rotation + scale). A naive nearest-neighbour transfer in
unaligned frames produces spatially wrong labels (the positive *rate* looks plausible but
the wrong *vertices* light up), so we:

1. Normalise the full splat means and the mesh vertices each to unit radius.
2. Rigid-ICP the normalised splat cloud onto the normalised mesh (this is the same
   alignment that ``notebooks/pipeline.ipynb`` performs by hand).
3. Map the annotation points through the same normalisation + ICP transform.
4. Label each mesh vertex 1 if an aligned annotation point lies within ``threshold``
   (in the shared unit-normalised frame).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from rendering.gaussian_ply import load_gaussian_splat_ply
from rendering.gsplat_viewpoint_selection import _scene_normalize_params, apply_scene_normalize


def _mesh_vertices(mesh_glb: Path) -> np.ndarray:
    loaded = trimesh.load(str(mesh_glb), process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise ValueError(f"No mesh geometry in {mesh_glb}")
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    else:
        mesh = loaded
    return np.asarray(mesh.vertices, dtype=np.float64)


def icp_rigid(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    n_iter: int = 60,
    n_samples: int = 4000,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid ICP (rotation + translation, no scale). Returns (aligned_src, 4x4 transform)."""
    rng = np.random.default_rng(rng_seed)
    tree = cKDTree(tgt)
    pts = src.astype(np.float64).copy()
    T_cum = np.eye(4)
    for _ in range(n_iter):
        idx = rng.choice(len(pts), min(n_samples, len(pts)), replace=False)
        query = pts[idx]
        _, nn = tree.query(query, k=1, workers=-1)
        tgt_q = tgt[nn]
        src_c, tgt_c = query.mean(0), tgt_q.mean(0)
        H = (query - src_c).T @ (tgt_q - tgt_c)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = tgt_c - R @ src_c
        pts = (R @ pts.T).T + t
        step = np.eye(4)
        step[:3, :3] = R
        step[:3, 3] = t
        T_cum = step @ T_cum
    return pts, T_cum


def affordance_labels_aligned(
    splat_ply: Path,
    anno_ply: Path,
    mesh_glb: Path,
    *,
    threshold: float = 0.05,
    n_iter: int = 60,
    return_debug: bool = False,
) -> np.ndarray | dict:
    """
    Binary per-vertex affordance labels for the SAM3D mesh, via normalise + ICP + NN transfer.

    Args:
        splat_ply:  original full Gaussian splat (alignment source — same frame as anno)
        anno_ply:   affordance-region annotation Gaussians
        mesh_glb:   SAM3D reconstructed mesh
        threshold:  max NN distance (unit-normalised frame) for a positive label

    Returns ``(V,)`` float32 labels, or a debug dict when ``return_debug`` is True.
    """
    splat_means = load_gaussian_splat_ply(splat_ply).means.astype(np.float64)
    anno_means = load_gaussian_splat_ply(anno_ply).means.astype(np.float64)
    verts = _mesh_vertices(mesh_glb)

    # Unit-normalise splat-frame (splat + anno share it) and mesh-frame separately.
    c_s, s_s = _scene_normalize_params(splat_means, target_radius=1.0)
    c_m, s_m = _scene_normalize_params(verts, target_radius=1.0)
    splat_n = apply_scene_normalize(splat_means, c_s, s_s)
    anno_n = apply_scene_normalize(anno_means, c_s, s_s)
    verts_n = apply_scene_normalize(verts, c_m, s_m)

    # Align the full splat cloud onto the mesh; reuse the transform for the anno points.
    _, T = icp_rigid(splat_n, verts_n, n_iter=n_iter)
    R, t = T[:3, :3], T[:3, 3]
    anno_aligned = (R @ anno_n.T).T + t

    dists, _ = cKDTree(anno_aligned).query(verts_n, k=1, workers=-1)
    labels = (dists < threshold).astype(np.float32)

    if not return_debug:
        return labels

    splat_aligned = (R @ splat_n.T).T + t
    d_fit, _ = cKDTree(verts_n).query(splat_aligned, k=1, workers=-1)
    return {
        "labels": labels,
        "positive_rate": float(labels.mean()),
        "icp_mean_residual": float(d_fit.mean()),
        "verts_n": verts_n,
        "anno_aligned": anno_aligned,
        "transform": T,
    }
