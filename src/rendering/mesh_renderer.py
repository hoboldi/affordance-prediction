from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyrender
import trimesh

from datasets.mesh_loading import MeshData
from rendering.camera_sampling import (
    CameraPose,
    resolve_spherical_orbit_axis,
    spherical_camera_poses,
)


@dataclass
class RenderView:
    """One rendered view with per-vertex 2D correspondences."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, 0 = background
    vertex_uv: np.ndarray  # (V, 2) float32, nan if not visible
    vertex_visible: np.ndarray  # (V,) bool
    camera_pose: np.ndarray  # (4, 4) camera-to-world
    intrinsics: np.ndarray  # (3, 3) pinhole K
    # Optional geometry channels for muddy / low-texture RGB (e.g. splats): SAM-friendly edges.
    normal_rgb: np.ndarray | None = None  # (H, W, 3) uint8 — mesh world normals as RGB
    depth_vis_rgb: np.ndarray | None = None  # (H, W, 3) uint8 — depth as 3× grayscale


@dataclass
class MeshRenderConfig:
    """Orbit camera layout: fixed pitch in ``[elevation_min_deg, elevation_max_deg]``, azimuth-only sweep."""

    image_size: int = 512
    fov_deg: float = 60.0
    num_views: int = 6
    camera_radius: float = 2.0
    elevation_deg: float = 42.5  # polar tilt from ring plane toward +orbit_axis; intersected with policy band
    elevation_min_deg: float = 25.0
    elevation_max_deg: float = 60.0
    # Orbit: cameras sweep a ring in the plane ⟂ ``orbit_axis`` (world units, same as mesh/splat).
    # ``ring_rotation_deg`` rotates the resolved pole (e.g. +90 about X when assets sit in a frame
    # tilted vs world +Y). ``None`` + ``orbit_axis_mode`` selects the pole before that rotation.
    orbit_axis: tuple[float, float, float] | None = None
    orbit_axis_mode: str = "world"
    orbit_ring_rotation_deg: float = 0.0
    orbit_ring_rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # If set, place cameras at these azimuths (degrees) on the ring instead of a uniform full sweep.
    # Length must match ``num_views`` (same zero azimuth as the first uniform sample).
    orbit_azimuth_offsets_deg: tuple[float, ...] | None = None
    depth_tolerance: float = 0.05
    depth_relative_tolerance: float = 0.03
    render_geometry_aux: bool = True  # world normal RGB + depth visualization (uint8)


class MeshRenderer:
    """Rasterize a mesh from multiple viewpoints (pyrender offscreen)."""

    def __init__(self, config: MeshRenderConfig | None = None) -> None:
        self.config = config or MeshRenderConfig()

    def _default_poses(self, mesh: MeshData) -> list[CameraPose]:
        ref = np.asarray(mesh.vertices, dtype=np.float64)
        axis = resolve_spherical_orbit_axis(
            ref,
            orbit_axis=self.config.orbit_axis,
            orbit_axis_mode=self.config.orbit_axis_mode,
            ring_rotation_deg=self.config.orbit_ring_rotation_deg,
            ring_rotation_axis=self.config.orbit_ring_rotation_axis,
        )
        return spherical_camera_poses(
            self.config.num_views,
            radius=self.config.camera_radius,
            elevation_deg=self.config.elevation_deg,
            elevation_min_deg=self.config.elevation_min_deg,
            elevation_max_deg=self.config.elevation_max_deg,
            orbit_axis=axis,
            azimuth_offsets_deg=self.config.orbit_azimuth_offsets_deg,
        )

    def render(
        self,
        mesh: MeshData,
        *,
        poses: list[CameraPose] | None = None,
    ) -> list[RenderView]:
        poses = poses or self._default_poses(mesh)
        trimesh_mesh = self._to_trimesh(mesh)
        vertex_normals = np.asarray(trimesh_mesh.vertex_normals, dtype=np.float64)
        py_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
        py_mesh_normal = (
            pyrender.Mesh.from_trimesh(_trimesh_vertex_normals_as_colors(trimesh_mesh), smooth=False)
            if self.config.render_geometry_aux
            else None
        )

        width = height = self.config.image_size
        yfov = np.deg2rad(self.config.fov_deg)
        aspect = width / height
        znear, zfar = 0.05, 10.0
        intrinsics = _perspective_intrinsics(width, height, yfov)

        renderer = pyrender.OffscreenRenderer(width, height)
        views: list[RenderView] = []

        try:
            for pose in poses:
                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.55, 0.55, 0.55, 1.0])
                scene.add(py_mesh)

                camera = pyrender.PerspectiveCamera(
                    yfov=yfov, aspectRatio=aspect, znear=znear, zfar=zfar
                )
                cam_node = scene.add(camera, pose=pose.matrix)

                light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
                scene.add(light, pose=pose.matrix)

                rgb, depth = renderer.render(scene)
                rgb = rgb[:, :, :3].astype(np.uint8)
                depth = depth.astype(np.float32)

                vertex_uv, vertex_visible = _vertex_correspondences(
                    mesh.vertices,
                    vertex_normals,
                    pose.matrix,
                    width,
                    height,
                    yfov,
                    znear,
                    zfar,
                    depth,
                    depth_tolerance=self.config.depth_tolerance,
                    depth_relative_tolerance=self.config.depth_relative_tolerance,
                )

                normal_rgb: np.ndarray | None = None
                depth_vis_rgb: np.ndarray | None = None
                if self.config.render_geometry_aux and py_mesh_normal is not None:
                    scene_n = pyrender.Scene(
                        bg_color=[0.06, 0.06, 0.08, 1.0],
                        ambient_light=[1.0, 1.0, 1.0, 1.0],
                    )
                    scene_n.add(py_mesh_normal)
                    camera_n = pyrender.PerspectiveCamera(
                        yfov=yfov, aspectRatio=aspect, znear=znear, zfar=zfar
                    )
                    cam_n2 = scene_n.add(camera_n, pose=pose.matrix)
                    nbuf, _ = renderer.render(scene_n)
                    normal_rgb = nbuf[:, :, :3].astype(np.uint8)
                    scene_n.remove_node(cam_n2)
                    depth_vis_rgb = _depth_buffer_to_vis_rgb(depth)

                views.append(
                    RenderView(
                        rgb=rgb,
                        depth=depth,
                        vertex_uv=vertex_uv,
                        vertex_visible=vertex_visible,
                        camera_pose=pose.matrix.astype(np.float64),
                        intrinsics=intrinsics,
                        normal_rgb=normal_rgb,
                        depth_vis_rgb=depth_vis_rgb,
                    )
                )
                scene.remove_node(cam_node)
        finally:
            renderer.delete()

        return views

    @staticmethod
    def _to_trimesh(mesh: MeshData) -> trimesh.Trimesh:
        vertex_colors = None
        if mesh.vertex_colors is not None:
            colors = mesh.vertex_colors
            if colors.shape[1] == 4:
                colors = colors[:, :3]
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            vertex_colors = colors

        return trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=vertex_colors,
            process=False,
        )


def _trimesh_vertex_normals_as_colors(tm: trimesh.Trimesh) -> trimesh.Trimesh:
    """Encode unit **world** mesh normals as RGB (0.5 * n + 0.5) for an unlit diagnostic pass."""
    vn = np.asarray(tm.vertex_normals, dtype=np.float64)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vn = vn / norms
    rgb = np.clip((vn * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
    return trimesh.Trimesh(
        vertices=tm.vertices,
        faces=tm.faces,
        vertex_colors=rgb,
        process=False,
    )


def _depth_buffer_to_vis_rgb(depth: np.ndarray, *, valid_eps: float = 1e-6) -> np.ndarray:
    """Map linear depth to uint8 RGB (3× grayscale) for aux inputs alongside muddy RGB."""
    d = np.asarray(depth, dtype=np.float64)
    mask = d > valid_eps
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not np.any(mask):
        return out
    vals = d[mask]
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if hi <= lo + 1e-8:
        hi = lo + 1e-6
    t = (d - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    t[~mask] = 0.0
    g = (t * 255.0 + 0.5).astype(np.uint8)
    out[..., 0] = g
    out[..., 1] = g
    out[..., 2] = g
    return out


def _perspective_intrinsics(width: int, height: int, yfov: float) -> np.ndarray:
    fy = height / (2.0 * np.tan(yfov / 2.0))
    fx = fy * (width / height)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _vertex_correspondences(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
    yfov: float,
    znear: float,
    zfar: float,
    depth_buffer: np.ndarray,
    *,
    depth_tolerance: float,
    depth_relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project vertices with the same OpenGL matrices as pyrender, then depth-test.
    """
    aspect = width / height
    camera = pyrender.PerspectiveCamera(
        yfov=yfov, aspectRatio=aspect, znear=znear, zfar=zfar
    )
    projection = camera.get_projection_matrix(width, height)
    view = np.linalg.inv(camera_to_world)

    ones = np.ones((len(vertices), 1), dtype=np.float64)
    world_h = np.concatenate([vertices.astype(np.float64), ones], axis=1)
    cam_h = (view @ world_h.T).T
    clip_h = (projection @ cam_h.T).T

    w = clip_h[:, 3]
    valid_w = np.abs(w) > 1e-8

    ndc = np.zeros((len(vertices), 3), dtype=np.float64)
    ndc[valid_w] = clip_h[valid_w, :3] / w[valid_w, None]

    in_frustum = (
        valid_w
        & (ndc[:, 0] >= -1.0)
        & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0)
        & (ndc[:, 1] <= 1.0)
        & (ndc[:, 2] >= -1.0)
        & (ndc[:, 2] <= 1.0)
    )

    # Camera looks down −Z; in-front points have negative camera-space Z.
    in_front = cam_h[:, 2] < -1e-6

    eye = camera_to_world[:3, 3]
    facing = np.einsum("ij,ij->i", vertex_normals, eye - vertices) > 0.0

    u = np.full(len(vertices), np.nan, dtype=np.float32)
    v = np.full(len(vertices), np.nan, dtype=np.float32)
    visible = np.zeros(len(vertices), dtype=bool)

    # NDC (+Y up) → image (+row down), matching flipped pyrender RGB/depth buffers.
    u_img = (ndc[:, 0] + 1.0) * 0.5 * width
    v_img = (1.0 - ndc[:, 1]) * 0.5 * height

    candidate = in_frustum & in_front & facing
    u[candidate] = u_img[candidate].astype(np.float32)
    v[candidate] = v_img[candidate].astype(np.float32)

    ui = np.round(u_img[candidate]).astype(int)
    vi = np.round(v_img[candidate]).astype(int)
    in_bounds = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)

    idx_candidate = np.where(candidate)[0][in_bounds]
    if len(idx_candidate) > 0:
        px, py = ui[in_bounds], vi[in_bounds]
        buf_z = depth_buffer[py, px]
        z_linear = -cam_h[idx_candidate, 2]
        tol = np.maximum(depth_tolerance, depth_relative_tolerance * z_linear)
        ok = (buf_z > 0) & (np.abs(z_linear - buf_z) <= tol)
        visible[idx_candidate[ok]] = True

    vertex_uv = np.stack([u, v], axis=1)
    return vertex_uv, visible
