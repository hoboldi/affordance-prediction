"""Resolve SAM3D mesh.glb for a specific splat via meta_prerender.json."""

from __future__ import annotations

import json
from pathlib import Path

from reconstruction.mesh_utils import find_sam3d_reconstruction_mesh_for_splat


def test_find_mesh_matches_meta_prerender(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    run = exports / "my_run"
    (run / "sam3d_dataset").mkdir(parents=True)
    (run / "reconstruction").mkdir(parents=True)

    ply = tmp_path / "data" / "Seen" / "train" / "shelf" / "Gaussian" / "GS_99.ply"
    ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write_text("dummy")

    meta = {"splat_path": str(ply.resolve()), "views": []}
    (run / "sam3d_dataset" / "meta_prerender.json").write_text(json.dumps(meta), encoding="utf-8")
    mesh = run / "reconstruction" / "mesh.glb"
    mesh.write_bytes(b"x")

    got = find_sam3d_reconstruction_mesh_for_splat(ply, search_under=tmp_path)
    assert got == mesh


def test_find_mesh_picks_newest_when_two_runs_match(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    ply = tmp_path / "data" / "Seen" / "train" / "bag" / "Gaussian" / "GS_1.ply"
    ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write_text("dummy")
    sp = str(ply.resolve())

    for name, age in (("old", 10), ("new", 20)):
        run = exports / name
        (run / "sam3d_dataset").mkdir(parents=True)
        (run / "reconstruction").mkdir(parents=True)
        (run / "sam3d_dataset" / "meta_prerender.json").write_text(
            json.dumps({"splat_path": sp, "views": []}), encoding="utf-8"
        )
        mesh = run / "reconstruction" / "mesh.glb"
        mesh.write_bytes(b"x")
        Path.utime(mesh, (age, age))

    got = find_sam3d_reconstruction_mesh_for_splat(ply, search_under=tmp_path)
    assert got.parent.parent.name == "new"


def test_find_mesh_none_when_meta_differs(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    run = exports / "other"
    (run / "sam3d_dataset").mkdir(parents=True)
    (run / "reconstruction").mkdir(parents=True)
    ply_a = tmp_path / "a.ply"
    ply_b = tmp_path / "b.ply"
    ply_a.write_text("x")
    ply_b.write_text("y")
    (run / "sam3d_dataset" / "meta_prerender.json").write_text(
        json.dumps({"splat_path": str(ply_a.resolve()), "views": []}), encoding="utf-8"
    )
    (run / "reconstruction" / "mesh.glb").write_bytes(b"z")

    assert find_sam3d_reconstruction_mesh_for_splat(ply_b, search_under=tmp_path) is None
