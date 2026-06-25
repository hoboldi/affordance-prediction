"""
Procedural **pitcher / watering-can-style** mesh for rendering demos.

No bundled GLB is required: trimesh primitives only. Use this when notebooks need a
recognisable object RGB instead of the synthetic Gaussian sphere PLY.
"""

from __future__ import annotations

import numpy as np
import trimesh

from datasets.mesh_loading import MeshData, normalize_mesh


def _mesh_data_from_trimesh(mesh: trimesh.Trimesh) -> MeshData:
    """Same contract as ``datasets.mesh_loading.mesh_data_from_trimesh`` (inlined for stale Docker trees)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_colors = None
    if mesh.visual is not None and hasattr(mesh.visual, "vertex_colors"):
        colors = mesh.visual.vertex_colors
        if colors is not None and len(colors) == len(vertices):
            vertex_colors = np.asarray(colors)
    return MeshData(vertices=vertices, faces=faces, vertex_colors=vertex_colors)


def build_demo_pitcher_mesh(*, normalize_scene: bool = True) -> MeshData:
    """
    Build a simple closed solid (body + neck + spout + handle) with vertex colours.

    This is **not** a photogrammetry asset — it is a stand-in so ``MeshRenderer`` produces
    object-like RGB/depth without shipping large mesh files.
    """
    parts: list[trimesh.Trimesh] = []

    body = trimesh.creation.cylinder(radius=0.22, height=0.36, sections=40)
    body.apply_translation([0.0, 0.18, 0.0])
    parts.append(body)

    neck = trimesh.creation.cylinder(radius=0.12, height=0.12, sections=28)
    neck.apply_translation([0.0, 0.45, 0.0])
    parts.append(neck)

    spout = trimesh.creation.cylinder(radius=0.034, height=0.34, sections=14)
    spout.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-18.0), [0, 0, 1]))
    spout.apply_translation([0.29, 0.42, 0.0])
    parts.append(spout)

    handle = trimesh.creation.cylinder(radius=0.028, height=0.28, sections=14)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0]))
    handle.apply_translation([-0.24, 0.34, 0.0])
    parts.append(handle)

    combined = trimesh.util.concatenate(parts)
    n = len(combined.vertices)
    terracotta = np.array([175, 88, 62, 255], dtype=np.uint8)
    vc = np.broadcast_to(terracotta, (n, 4)).copy()
    combined.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)

    mesh = _mesh_data_from_trimesh(combined)
    if normalize_scene:
        mesh = normalize_mesh(mesh, target_extent=0.95)
    return mesh
