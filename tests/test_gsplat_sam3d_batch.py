"""Unit tests for ``reconstruction.gsplat_sam3d_batch`` (no SAM3D / CUDA)."""

from __future__ import annotations

import json
from pathlib import Path

from reconstruction.gsplat_sam3d_batch import (
    ensure_sam3d_reconstruction_for_splat,
    sam3d_run_layout_ready,
    splat_run_dir_slug,
)


def test_splat_run_dir_slug_seen_train() -> None:
    p = Path("/data/Seen/train/bag/Gaussian/GS_0017.ply")
    assert splat_run_dir_slug(p) == "Seen_train_bag_Gaussian_GS_0017"


def test_sam3d_run_layout_ready_requires_mesh_and_meta() -> None:
    run = Path("/tmp/nonexistent_run_xyz")
    assert not sam3d_run_layout_ready(run)


def test_ensure_skips_when_layout_matches(tmp_path: Path) -> None:
    ply = tmp_path / "mirror" / "Seen" / "train" / "bag" / "Gaussian" / "GS_7.ply"
    ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write_text("ply")

    name = "Seen_train_bag_Gaussian_GS_7"
    run = tmp_path / "out" / name
    (run / "sam3d_dataset").mkdir(parents=True)
    (run / "reconstruction").mkdir(parents=True)
    (run / "reconstruction" / "mesh.glb").write_bytes(b"glb")
    (run / "sam3d_dataset" / "meta_prerender.json").write_text(
        json.dumps({"splat_path": str(ply.resolve()), "views": []}),
        encoding="utf-8",
    )

    r = ensure_sam3d_reconstruction_for_splat(
        ply,
        output_root=tmp_path / "out",
        cfg={},
        run_dir_name=name,
    )
    assert r["status"] == "skipped"
    assert r["mesh_glb"] == run / "reconstruction" / "mesh.glb"
