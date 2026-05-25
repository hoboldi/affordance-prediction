"""
Render multi-view RGB + depth from a 3DGS ``.ply`` by drawing **Gaussian centers** as coloured points.

This is a **preview / SAM3D-input generator**, not full ellipsoidal 3DGS rasterization. For publication-quality
splatting, integrate a CUDA rasterizer (e.g. gsplat) separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyrender
import trimesh

from rendering.camera_sampling import spherical_camera_poses
from rendering.gaussian_ply import GaussianSplatPLY, load_gaussian_splat_ply, sh_dc_to_rgb, sigmoid
from rendering.mesh_renderer import MeshRenderConfig, RenderView


def _perspective_intrinsics(width: int, height: int, yfov: float) -> np.ndarray:
    fy = height / (2.0 * np.tan(yfov / 2.0))
    fx = fy * (width / height)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _normalize_points(points: np.ndarray, *, target_radius: float = 0.9) -> np.ndarray:
    """Center at origin and isotropically scale so max extent fits in ``target_radius``."""
    c = points.mean(axis=0)
    centered = points - c
    extent = np.linalg.norm(centered, axis=1).max()
    if extent < 1e-8:
        return centered.astype(np.float32)
    return (centered * (target_radius / extent)).astype(np.float32)


def _radial_normals(vertices: np.ndarray) -> np.ndarray:
    """Unit normals aligned with position (outward on a centered cloud) for PBR lighting."""
    v = np.asarray(vertices, dtype=np.float64)
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.divide(v, np.clip(nrm, 1e-8, None))
    return out.astype(np.float32)


def _merge_icosphere_blobs(vertices: np.ndarray, colors01: np.ndarray, *, sphere_radius: float) -> pyrender.Mesh:
    """
    One low-poly icosphere per Gaussian centre — visible under pyrender lighting (unlike ``GL_POINTS``).
    """
    v = np.asarray(vertices, dtype=np.float32)
    rgb_u8 = (np.clip(colors01[:, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    base = trimesh.creation.icosphere(subdivisions=0, radius=float(sphere_radius))
    pieces: list[trimesh.Trimesh] = []
    n_verts = len(base.vertices)
    for i in range(len(v)):
        g = base.copy()
        g.apply_translation(v[i])
        vc = np.zeros((n_verts, 4), dtype=np.uint8)
        vc[:, 3] = 255
        vc[:, :3] = rgb_u8[i]
        g.visual.vertex_colors = vc
        pieces.append(g)
    combined = trimesh.util.concatenate(pieces)
    return pyrender.Mesh.from_trimesh(combined, smooth=True)


def _point_sprites_mesh(vertices: np.ndarray, colors01: np.ndarray) -> pyrender.Mesh:
    """
    ``GL_POINTS`` with vertex colours + outward normals and a neutral PBR material.

    Many drivers still render 1 px points; prefer :func:`_merge_icosphere_blobs` when ``N`` is small.
    """
    v = np.asarray(vertices, dtype=np.float32)
    c = np.clip(colors01, 0.0, 1.0).astype(np.float32)
    rgba = np.concatenate([c[:, :3], np.ones((len(v), 1), dtype=np.float32)], axis=1)
    normals = _radial_normals(v)
    mesh = pyrender.Mesh.from_points(v, colors=rgba, normals=normals)
    mat = pyrender.MetallicRoughnessMaterial(
        alphaMode="OPAQUE",
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.85,
    )
    for prim in mesh.primitives:
        prim.material = mat
    return mesh


def _preview_gaussian_mesh(
    vertices: np.ndarray,
    colors01: np.ndarray,
    *,
    preview_mode: str,
    sphere_merge_max: int,
    sphere_radius: float | None,
) -> pyrender.Mesh:
    v = np.asarray(vertices, dtype=np.float32)
    n = len(v)
    mode = preview_mode.lower().strip()
    if mode == "auto":
        mode = "spheres" if n <= sphere_merge_max else "points"
    if mode == "spheres":
        if sphere_radius is None:
            extent = float(np.max(np.linalg.norm(v, axis=1))) + 1e-6
            sr = float(np.clip(0.045 * extent, 0.006, 0.08))
        else:
            sr = float(sphere_radius)
        return _merge_icosphere_blobs(v, colors01, sphere_radius=sr)
    if mode == "points":
        return _point_sprites_mesh(v, colors01)
    raise ValueError(f"preview_mode must be auto|spheres|points, got {preview_mode!r}")


def splat_ply_to_point_cloud(
    ply: GaussianSplatPLY | Path | str,
    *,
    max_points: int = 200_000,
    seed: int = 0,
    normalize_scene: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert splat PLY to point positions and RGB (``[0,1]``).

    Subsamples uniformly when ``N > max_points``.
    """
    if not isinstance(ply, GaussianSplatPLY):
        ply = load_gaussian_splat_ply(Path(ply))

    means = ply.means.astype(np.float32)
    rgb = sh_dc_to_rgb(ply.f_dc)
    alpha = sigmoid(ply.opacity.astype(np.float64)).astype(np.float32)
    rgb = rgb * alpha[:, None] + (1.0 - alpha[:, None]) * 0.15  # dim background-ish points

    n = means.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        means = means[idx]
        rgb = rgb[idx]

    if normalize_scene:
        means = _normalize_points(means, target_radius=0.95)

    return means.astype(np.float32), np.clip(rgb, 0.0, 1.0).astype(np.float32)


def render_gaussian_splat_views(
    ply_path: str | Path,
    config: MeshRenderConfig | None = None,
    *,
    max_points: int = 200_000,
    normalize_scene: bool = True,
    seed: int = 0,
    preview_mode: str = "auto",
    sphere_merge_max: int = 2048,
    sphere_radius: float | None = None,
) -> list[RenderView]:
    """
    Render ``num_views`` around the splat centroid using the same camera layout as ``MeshRenderer``.

    ``preview_mode``:
        - ``auto`` (default): merged **icosphere** blobs when ``N <= sphere_merge_max`` (visible
          under stock pyrender lighting); otherwise ``GL_POINTS`` with fake normals.
        - ``spheres`` / ``points``: force that path.

    Returns :class:`rendering.mesh_renderer.RenderView` entries with ``vertex_uv`` all NaN and
    ``vertex_visible`` empty — point splats have no mesh vertex correspondences.
    """
    cfg = config or MeshRenderConfig()
    ply = load_gaussian_splat_ply(Path(ply_path))
    means, rgb = splat_ply_to_point_cloud(
        ply, max_points=max_points, seed=seed, normalize_scene=normalize_scene
    )

    poses = spherical_camera_poses(
        cfg.num_views,
        radius=cfg.camera_radius,
        elevation_deg=cfg.elevation_deg,
    )

    width = height = cfg.image_size
    yfov = np.deg2rad(cfg.fov_deg)
    aspect = width / height
    znear, zfar = 0.05, 10.0
    intrinsics = _perspective_intrinsics(width, height, yfov)

    py_mesh = _preview_gaussian_mesh(
        means,
        rgb,
        preview_mode=preview_mode,
        sphere_merge_max=sphere_merge_max,
        sphere_radius=sphere_radius,
    )
    renderer = pyrender.OffscreenRenderer(width, height)
    views: list[RenderView] = []
    v_empty = np.zeros((0, 2), dtype=np.float32)
    vis_empty = np.zeros((0,), dtype=bool)

    try:
        for pose in poses:
            scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.14, 1.0], ambient_light=[0.75, 0.75, 0.75, 1.0])
            scene.add(py_mesh)

            camera = pyrender.PerspectiveCamera(
                yfov=yfov, aspectRatio=aspect, znear=znear, zfar=zfar
            )
            cam_node = scene.add(camera, pose=pose.matrix)

            light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
            scene.add(light, pose=pose.matrix)

            rgb_img, depth = renderer.render(scene)
            rgb_img = rgb_img[:, :, :3].astype(np.uint8)
            depth = depth.astype(np.float32)

            views.append(
                RenderView(
                    rgb=rgb_img,
                    depth=depth,
                    vertex_uv=v_empty,
                    vertex_visible=vis_empty,
                    camera_pose=pose.matrix.astype(np.float64),
                    intrinsics=intrinsics,
                )
            )
            scene.remove_node(cam_node)
    finally:
        renderer.delete()

    return views
