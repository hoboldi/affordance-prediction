"""cfg_with_sam3d_reconstruction wiring for downstream notebooks."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from reconstruction.mesh_utils import cfg_with_sam3d_reconstruction


def test_cfg_with_sam3d_reconstruction_missing_mesh(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "reconstruction").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="SAM3D mesh not found"):
        cfg_with_sam3d_reconstruction({"rendering": {"backend": "mesh"}}, run)


def test_cfg_with_sam3d_reconstruction_sets_absolute_mesh_path(tmp_path: Path) -> None:
    run = tmp_path / "run"
    rec = run / "reconstruction"
    rec.mkdir(parents=True)
    trimesh.creation.box().export(rec / "mesh.glb")

    cfg = {"rendering": {"mesh_path": "data/sample.glb", "backend": "mesh", "splat_path": None}}
    out = cfg_with_sam3d_reconstruction(cfg, run, link_gaussian_splat=False)
    assert Path(out["rendering"]["mesh_path"]).resolve() == (rec / "mesh.glb").resolve()
    assert out["rendering"]["splat_path"] is None


def test_cfg_with_sam3d_reconstruction_sets_gaussian_backend_when_ply_present(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    rec = run / "reconstruction"
    rec.mkdir(parents=True)
    trimesh.creation.box().export(rec / "mesh.glb")
    (rec / "gaussian.ply").write_text("dummy")

    cfg = {"rendering": {"mesh_path": "data/sample.glb", "backend": "mesh", "splat_path": None}}
    out = cfg_with_sam3d_reconstruction(cfg, run)
    assert Path(out["rendering"]["splat_path"]).resolve() == (rec / "gaussian.ply").resolve()
    assert out["rendering"]["backend"] == "gaussian"


def test_cfg_with_sam3d_reconstruction_mesh_only_downgrades_gaussian_backend(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    rec = run / "reconstruction"
    rec.mkdir(parents=True)
    trimesh.creation.box().export(rec / "mesh.glb")

    cfg = {
        "rendering": {
            "mesh_path": "data/sample.glb",
            "backend": "gaussian",
            "splat_path": None,
        }
    }
    out = cfg_with_sam3d_reconstruction(cfg, run)
    assert out["rendering"]["splat_path"] is None
    assert out["rendering"]["backend"] == "mesh"
