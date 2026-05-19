from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyrender
import trimesh

from datasets.mesh_loading import MeshData
from rendering.camera_sampling import CameraPose, spherical_camera_poses


@dataclass
class RenderView:
    """One rendered view with per-vertex 2D correspondences."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, 0 = background
    vertex_uv: np.ndarray  # (V, 2) float32, nan if not visible
    vertex_visible: np.ndarray  # (V,) bool
    camera_pose: np.ndarray  # (4, 4) camera-to-world
    intrinsics: np.ndarray  # (3, 3) pinhole K


@dataclass
class MeshRenderConfig:
    image_size: int = 512
    fov_deg: float = 60.0
    num_views: int = 6
    camera_radius: float = 2.0
    elevation_deg: float = 30.0
    depth_tolerance: float = 0.05
    depth_relative_tolerance: float = 0.03


class MeshRenderer:
    """Rasterize a mesh from multiple viewpoints (pyrender offscreen)."""

    def __init__(self, config: MeshRenderConfig | None = None) -> None:
        self.config = config or MeshRenderConfig()

    def render(
        self,
        mesh: MeshData,
        *,
        poses: list[CameraPose] | None = None,
    ) -> list[RenderView]:
        poses = poses or spherical_camera_poses(
            self.config.num_views,
            radius=self.config.camera_radius,
            elevation_deg=self.config.elevation_deg,
        )
        trimesh_mesh = self._to_trimesh(mesh)
        vertex_normals = np.asarray(trimesh_mesh.vertex_normals, dtype=np.float64)
        py_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)

        width = height = self.config.image_size
        yfov = np.deg2rad(self.config.fov_deg)
        aspect = width / height
        znear, zfar = 0.05, 10.0
        intrinsics = _perspective_intrinsics(width, height, yfov)

        renderer = pyrender.OffscreenRenderer(width, height)
        views: list[RenderView] = []

        try:
            for pose in poses:
                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.45, 0.45, 0.45, 1.0])
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

                views.append(
                    RenderView(
                        rgb=rgb,
                        depth=depth,
                        vertex_uv=vertex_uv,
                        vertex_visible=vertex_visible,
                        camera_pose=pose.matrix.astype(np.float64),
                        intrinsics=intrinsics,
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
