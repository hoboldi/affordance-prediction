from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh

MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply", ".stl"}


@dataclass
class MeshData:
    """Triangle mesh in world coordinates."""

    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int64
    vertex_colors: np.ndarray | None = None  # (V, 3|4) uint8 or float

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def center(self) -> np.ndarray:
        return self.vertices.mean(axis=0)


def load_mesh(path: str | Path, *, process: bool = True) -> MeshData:
    """
    Load a mesh from GLB/OBJ/PLY/STL.

    Multi-geometry GLB scenes are merged into a single mesh.
    """
    path = Path(path)
    if path.suffix.lower() not in MESH_EXTENSIONS:
        raise ValueError(f"Unsupported mesh format: {path.suffix}")

    loaded = trimesh.load(path, force="mesh", process=process)
    mesh = _to_single_trimesh(loaded, path)
    return _trimesh_to_mesh_data(mesh)


def mesh_data_from_trimesh(mesh: trimesh.Trimesh) -> MeshData:
    """Convert a single ``trimesh.Trimesh`` (already merged / processed) to :class:`MeshData`."""
    return _trimesh_to_mesh_data(mesh)


def normalize_mesh(
    mesh: MeshData,
    *,
    center: bool = True,
    unit_scale: bool = True,
    target_extent: float = 1.0,
) -> MeshData:
    """
    Center mesh at the origin and optionally scale to ``target_extent``.

    Scaling uses the maximum distance from the origin after centering.
    """
    vertices = mesh.vertices.astype(np.float64, copy=True)

    if center:
        vertices -= mesh.center()

    if unit_scale:
        radius = np.linalg.norm(vertices, axis=1).max()
        if radius > 0:
            vertices = vertices / radius * target_extent

    return replace(mesh, vertices=vertices.astype(np.float32))


def _to_single_trimesh(loaded: trimesh.Trimesh | trimesh.Scene, path: Path) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        if len(geometries) == 1:
            return geometries[0]
        return trimesh.util.concatenate(geometries)

    raise TypeError(f"Unexpected trimesh load result for {path}: {type(loaded)}")


def _trimesh_to_mesh_data(mesh: trimesh.Trimesh) -> MeshData:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    vertex_colors = None
    if mesh.visual is not None and hasattr(mesh.visual, "vertex_colors"):
        colors = mesh.visual.vertex_colors
        if colors is not None and len(colors) == len(vertices):
            vertex_colors = np.asarray(colors)

    return MeshData(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
