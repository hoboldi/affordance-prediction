"""
Idempotent **gsplat → SAM3D** batching for pipeline use and notebooks.

Each splat gets a deterministic run directory under ``output_root`` (slug from ``Seen/…`` or
``UnSeen/…`` path). If ``reconstruction/mesh.glb`` + ``sam3d_dataset/meta_prerender.json`` already
exist and the prerender metadata matches the splat, the run is **skipped** unless ``force=True``.

**Requires** the ``sam3d-pipeline`` Docker image (CUDA + gsplat + SAM3D); see ``docker/README.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from reconstruction.gsplat_to_sam3d import gsplat_ply_to_sam3d_reconstruction
from reconstruction.mesh_utils import prerender_meta_matches_splat
from reconstruction.sam3d_wrapper import SAM3DWrapper
from utils.config import resolve_path


def splat_run_dir_slug(splat_ply: str | Path) -> Path:
    """
    Stable nested path under the output root from an AffordSplat-style splat path.

    Mirrors the source directory structure so reconstructions are easy to locate:
      ``…/Seen/train/bag/Gaussian/GS_0017.ply`` → ``Seen/train/bag/Gaussian/GS_0017``

    Falls back to a flat safe name for paths that don't follow the AffordSplat layout.
    """
    p = Path(splat_ply).expanduser().resolve()
    lower = [x.lower() for x in p.parts]
    for marker in ("seen", "unseen"):
        if marker in lower:
            i = lower.index(marker)
            return Path(*p.parts[i:]).with_suffix("")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in p.name)
    return Path(f"splat_{safe}"[:220])


def sam3d_run_layout_ready(run_dir: Path) -> bool:
    """Whether a run directory looks like a finished gsplat→SAM3D export."""
    rd = Path(run_dir).expanduser().resolve()
    recon = rd / "reconstruction"
    return (
        (recon / "mesh.glb").is_file()
        and (recon / "shape_latent.pt").is_file()
        and (recon / "slat_feats.pt").is_file()
        and (recon / "slat_coords.pt").is_file()
        and (recon / "slat_vertex_features.pt").is_file()
        and (recon / "global_latent.pt").is_file()
        and (recon / "dino_cls.pt").is_file()
        and (recon / "dino_patches.pt").is_file()
        and (recon / "ss_dino_cls.pt").is_file()
        and (recon / "ss_dino_patches.pt").is_file()
        and (recon / "vertex_normals.pt").is_file()
        and (rd / "sam3d_dataset" / "meta_prerender.json").is_file()
    )


def ensure_sam3d_reconstruction_for_splat(
    splat_ply: str | Path,
    *,
    output_root: str | Path,
    cfg: dict[str, Any] | None = None,
    wrapper: SAM3DWrapper | None = None,
    force: bool = False,
    run_dir_name: str | Path | None = None,
    **recon_kw: Any,
) -> dict[str, Any]:
    """
    Run :func:`reconstruction.gsplat_to_sam3d.gsplat_ply_to_sam3d_reconstruction` once for ``splat_ply``.

    Pass a pre-loaded ``wrapper`` to avoid reloading the SAM3D model on every call (required for
    batch processing — see :func:`batch_ensure_sam3d_reconstructions`).

    Parameters mirror ``gsplat_ply_to_sam3d_reconstruction`` (``max_points``, ``reference_view_index``,
    ``sam3d_seed``, ``gsplat_seed``, …) via ``**recon_kw``.

    Returns a dict with ``status`` in ``{"ran", "skipped"}`` plus ``run_dir``, ``mesh_glb``, etc.
    """
    splat = resolve_path(splat_ply)
    if not splat.is_file():
        raise FileNotFoundError(splat)

    out_root = Path(resolve_path(output_root))
    out_root.mkdir(parents=True, exist_ok=True)
    name = run_dir_name or splat_run_dir_slug(splat)
    run_dir = (out_root / name).resolve()
    meta_path = run_dir / "sam3d_dataset" / "meta_prerender.json"
    mesh_path = run_dir / "reconstruction" / "mesh.glb"

    if (
        not force
        and sam3d_run_layout_ready(run_dir)
        and prerender_meta_matches_splat(meta_path, splat)
    ):
        return {
            "status": "skipped",
            "splat_path": splat,
            "run_dir": run_dir,
            "reconstruction_dir": run_dir / "reconstruction",
            "mesh_glb": mesh_path,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    out = gsplat_ply_to_sam3d_reconstruction(splat, run_dir, cfg=cfg, wrapper=wrapper, **recon_kw)
    recon_dir = Path(out["reconstruction_dir"])
    return {
        "status": "ran",
        "splat_path": splat,
        "run_dir": recon_dir.parent,
        "reconstruction_dir": recon_dir,
        "mesh_glb": recon_dir / "mesh.glb",
        "object_stem": out.get("object_stem"),
    }


def batch_ensure_sam3d_reconstructions(
    splat_paths: Iterable[str | Path],
    *,
    output_root: str | Path,
    cfg: dict[str, Any] | None = None,
    wrapper: SAM3DWrapper | None = None,
    force: bool = False,
    **recon_kw: Any,
) -> list[dict[str, Any]]:
    """Call :func:`ensure_sam3d_reconstruction_for_splat` for each path; order is preserved."""
    results: list[dict[str, Any]] = []
    for sp in splat_paths:
        results.append(
            ensure_sam3d_reconstruction_for_splat(
                sp, output_root=output_root, cfg=cfg, wrapper=wrapper, force=force, **recon_kw
            )
        )
    return results
