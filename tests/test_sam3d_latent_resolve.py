"""Resolve SAM3D ``global_latent.pt`` cache paths (see ``reconstruction.sam3d_wrapper``)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from reconstruction.sam3d_wrapper import (
    latent_cache_path,
    resolve_global_latent_cache_path,
    try_load_cached_global_latent,
)


def test_resolve_global_latent_via_object_stem(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    stem = "obj_a"
    p = latent_cache_path(stem, cache_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"global_latent": torch.zeros(8)}, p)

    cfg = {"paths": {"cache_root": str(cache_root)}}
    out = resolve_global_latent_cache_path(cfg, object_stem=stem, project_root_path=tmp_path)
    assert out == p

    z, zp = try_load_cached_global_latent(cfg, object_stem=stem, project_root_path=tmp_path)
    assert zp == p
    assert z is not None and z.shape == (8,)


def test_resolve_global_latent_via_run_meta(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    run = tmp_path / "run1"
    (run / "reconstruction").mkdir(parents=True)
    (run / "reconstruction" / "meta.json").write_text(json.dumps({"stem": "from_meta"}))

    p = latent_cache_path("from_meta", cache_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"global_latent": torch.ones(8)}, p)

    cfg = {"paths": {"cache_root": str(cache_root)}}
    resolved = resolve_global_latent_cache_path(cfg, sam3d_run_dir=run, project_root_path=tmp_path)
    assert resolved == p

    z, zp = try_load_cached_global_latent(cfg, sam3d_run_dir=run, project_root_path=tmp_path)
    assert z is not None and torch.allclose(z, torch.ones(8))


def test_try_load_returns_path_when_missing(tmp_path: Path) -> None:
    cfg = {"paths": {"cache_root": str(tmp_path / "cache")}}
    z, p = try_load_cached_global_latent(cfg, object_stem="nope", project_root_path=tmp_path)
    assert z is None
    assert p is not None and not p.is_file()
