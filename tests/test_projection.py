import numpy as np
import torch

from projection.multiview_fusion import fuse_patch_features
from projection.patch_vertex_mapping import render_uv_to_patch_index
from projection.project_to_mesh import ViewProjectionInputs, project_views_to_vertices
from vlm.patch_extractor import PatchFeatures


def _dummy_patch_features(value: float, grid: int = 7, dim: int = 8) -> PatchFeatures:
    n = grid * grid
    patches = torch.full((n, dim), value)
    return PatchFeatures(patches=patches, grid_h=grid, grid_w=grid, feature_dim=dim)


def test_visible_vertices_span_multiple_image_rows():
    """Regression: projection must not collapse visibility to a single scanline."""
    from datasets.mesh_loading import load_mesh, normalize_mesh
    from pathlib import Path
    from rendering.mesh_renderer import MeshRenderer

    sample = Path(__file__).resolve().parents[1] / "data" / "sample.glb"
    if not sample.exists():
        return
    mesh = normalize_mesh(load_mesh(sample))
    view = MeshRenderer().render(mesh)[0]
    vis = view.vertex_visible
    if vis.sum() < 10:
        return
    rows = np.unique(np.round(view.vertex_uv[vis, 1]).astype(int))
    assert len(rows) > 10, f"visible rows={len(rows)} (expected 2D coverage)"


def test_render_uv_to_patch_index_center():
    uv = np.array([[256.0, 256.0]], dtype=np.float32)
    vis = np.array([True])
    idx = render_uv_to_patch_index(
        uv,
        vis,
        render_height=512,
        render_width=512,
        clip_image_size=224,
        grid_h=7,
        grid_w=7,
    )
    assert idx[0] == 24  # center patch in 7x7 grid


def test_multiview_fusion_averages():
    v = 2
    d = 4
    patches_a = torch.ones(49, d)
    patches_b = torch.zeros(49, d)
    idx_a = np.array([10, -1], dtype=np.int64)
    idx_b = np.array([-1, 20], dtype=np.int64)

    fused, counts = fuse_patch_features(v, [idx_a, idx_b], [patches_a, patches_b])
    assert counts[0] == 1.0 and counts[1] == 1.0
    assert torch.allclose(fused[0], patches_a[10])
    assert torch.allclose(fused[1], patches_b[20])


def test_project_views_to_vertices():
    uv = np.zeros((3, 2), dtype=np.float32)
    uv[0] = [256, 256]
    uv[1] = [128, 128]
    uv[2] = [400, 400]
    vis = np.array([True, True, False])

    view = ViewProjectionInputs(
        vertex_uv=uv,
        vertex_visible=vis,
        patches=_dummy_patch_features(1.0),
        render_height=512,
        render_width=512,
    )
    out = project_views_to_vertices([view])
    assert out.features.shape == (3, 8)
    assert out.visible_in_any_view[0]
    assert out.visible_in_any_view[1]
    assert not out.visible_in_any_view[2]
