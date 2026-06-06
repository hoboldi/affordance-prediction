from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from utils.config import project_root, resolve_path


def resolve_sam3d_objects_root(project_root_path: Path | None = None) -> Path:
    """
    Locate the **facebookresearch/sam-3d-objects** checkout where ``notebook/inference.py`` lives.

    Search order:

    1. ``SAM3D_OBJECTS_ROOT`` or ``SAM3D_ROOT`` (absolute path to repo root)
    2. ``<project_root>/sam-3d-objects``
    3. ``<project_root>/../sam-3d-objects`` (sibling clone)

    Raises ``FileNotFoundError`` with a short checklist if nothing matches.
    """
    root = project_root_path or project_root()
    tried: list[str] = []

    for env_key in ("SAM3D_OBJECTS_ROOT", "SAM3D_ROOT"):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        tried.append(f"{env_key}={p}")
        if (p / "notebook" / "inference.py").is_file():
            return p

    for label, p in (
        ("<project_root>/sam-3d-objects", root / "sam-3d-objects"),
        ("<project_root>/../sam-3d-objects", root.parent / "sam-3d-objects"),
    ):
        tried.append(f"{label} -> {p}")
        if (p / "notebook" / "inference.py").is_file():
            return p

    lines = "\n  - ".join(tried) if tried else "(no env override; checked default paths)"
    raise FileNotFoundError(
        "SAM3D code not found (need notebook/inference.py inside sam-3d-objects). Checked:\n  - "
        f"{lines}\n"
        "Fix: run `git submodule update --init sam-3d-objects`, or clone "
        "https://github.com/facebookresearch/sam-3d-objects next to this repo, "
        "or set SAM3D_OBJECTS_ROOT to that checkout. "
        "Docker: image bundles under /workspace/sam-3d-objects; bind-mount must not hide it with an empty folder."
    )


@dataclass
class ReconstructionResult:
    """Outputs from a single SAM3D forward pass."""

    shape_latent: torch.Tensor        # (4096, 8) dense 16³ SS voxel grid
    global_latent: torch.Tensor       # (8,) mean-pooled SLAT features over occupied voxels
    slat_feats: torch.Tensor          # (N, 8) structured latent per occupied voxel
    slat_coords: torch.Tensor         # (N, 3) voxel coordinates
    dino_cls: torch.Tensor | None = None          # (768,) DINOv2 CLS token — SLAT embedder
    dino_patches: torch.Tensor | None = None      # (N_patches, 768) spatial patch tokens — SLAT embedder
    ss_dino_cls: torch.Tensor | None = None       # (768,) DINOv2 CLS token — SS embedder
    ss_dino_patches: torch.Tensor | None = None   # (N_patches, 768) spatial patch tokens — SS embedder
    gaussian_splat: Any | None = None
    mesh_scene: Any | None = None


@dataclass(frozen=True)
class _Sam3DEnvPatch:
    """State to undo :func:`_apply_sam3d_environment` mutations (``sys.path`` + ``LIDRA_SKIP_INIT``)."""

    inserted_path_strs: tuple[str, ...]
    lidra_skip_init_before: str | None


def _apply_sam3d_environment(root: Path | None = None) -> tuple[Path, _Sam3DEnvPatch]:
    root = root or project_root()
    sam3d_root = resolve_sam3d_objects_root(root)
    notebook_dir = sam3d_root / "notebook"

    inserted: list[str] = []
    for path in (sam3d_root, notebook_dir):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted.append(path_str)

    os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
    lidra_before = os.environ.get("LIDRA_SKIP_INIT")
    os.environ["LIDRA_SKIP_INIT"] = "true"
    return sam3d_root, _Sam3DEnvPatch(
        inserted_path_strs=tuple(inserted),
        lidra_skip_init_before=lidra_before,
    )


def _restore_sam3d_environment(patch: _Sam3DEnvPatch) -> None:
    """
    Remove inserted ``sys.path`` entries and restore ``LIDRA_SKIP_INIT``.

    ``CUDA_HOME`` is left as set by :func:`_apply_sam3d_environment` (typically defaulted from
    ``CONDA_PREFIX``) so CUDA tooling stays stable after the cell.
    """
    for path_str in reversed(patch.inserted_path_strs):
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass
    if patch.lidra_skip_init_before is None:
        os.environ.pop("LIDRA_SKIP_INIT", None)
    else:
        os.environ["LIDRA_SKIP_INIT"] = patch.lidra_skip_init_before


def configure_sam3d_environment(root: Path | None = None) -> Path:
    """
    Add sam-3d-objects to ``sys.path`` and set env vars required by the submodule.

    This leaves ``sys.path`` and ``os.environ`` changed for the rest of the process.
    Prefer :func:`sam3d_environment` in notebooks so later imports in the same cell
    still resolve against the project.

    Returns the sam-3d-objects directory path.
    """
    sam3d_root, _patch = _apply_sam3d_environment(root)
    return sam3d_root


@contextmanager
def sam3d_environment(root: Path | None = None) -> Iterator[Path]:
    """
    Prepend ``sam-3d-objects`` on ``sys.path`` and set SAM3D env vars for the block, then
    on exit **remove those ``sys.path`` entries** and restore ``LIDRA_SKIP_INIT`` to its prior
    value (SAM3D only needs it while the submodule loads).

    ``CUDA_HOME`` is still defaulted from ``CONDA_PREFIX`` when unset and is **not** removed
    afterward so CUDA tooling does not lose that hint after the cell.
    """
    sam3d_root, patch = _apply_sam3d_environment(root)
    try:
        yield sam3d_root
    finally:
        _restore_sam3d_environment(patch)


def _require_sam3d_pipeline_file(path: Path, *, sam3d_repo_root: Path) -> None:
    """Raise a clear error when ``pipeline.yaml`` (HF weights) is missing."""
    if path.is_file():
        return
    ck_root = sam3d_repo_root / "checkpoints"
    raise FileNotFoundError(
        f"SAM3D pipeline config not found: {path}\n"
        "Model weights are not in git; download them after Hugging Face access is granted for "
        "facebook/sam-3d-objects, then authenticate (e.g. `hf auth login` or HF_TOKEN). "
        "From the sam-3d-objects repository root, follow doc/setup.md section 2, e.g.:\n"
        "  pip install 'huggingface-hub[cli]<1.0'\n"
        "  TAG=hf && hf download --repo-type model --local-dir checkpoints/${TAG}-download "
        "--max-workers 1 facebook/sam-3d-objects\n"
        "  mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG} && rm -rf checkpoints/${TAG}-download\n"
        f"Expected directory after download: {ck_root / 'hf'}. "
        "Override path via reconstruction.sam3d_config in your YAML if you store weights elsewhere."
    )


class SAM3DWrapper:
    """
    Thin wrapper around the SAM3D ``Inference`` class.

    Construct this only while :func:`sam3d_environment` is active (or after a one-off
    :func:`configure_sam3d_environment` call), so ``from inference import Inference`` resolves.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        compile_model: bool = False,
        project_root_path: Path | None = None,
    ) -> None:
        root = project_root_path or project_root()

        try:
            from inference import Inference  # noqa: E402
        except ModuleNotFoundError as exc:
            if exc.name == "inference":
                sam = resolve_sam3d_objects_root(root)
                raise ModuleNotFoundError(
                    "Python could not import `inference` even though "
                    f"{sam / 'notebook' / 'inference.py'} exists — use "
                    "`with sam3d_environment():` (or `configure_sam3d_environment()` once), "
                    "the same interpreter/env as SAM3D (e.g. conda env `sam3d` in Docker), "
                    "and a correct ``sys.path``. Original error: "
                    f"{exc}"
                ) from exc
            raise

        resolved = resolve_path(config_path, root=root)
        sam_repo = resolve_sam3d_objects_root(root)
        _require_sam3d_pipeline_file(resolved, sam3d_repo_root=sam_repo)
        self._inference = Inference(str(resolved), compile=compile_model)
        self.config_path = resolved

    @property
    def pipeline(self) -> Any:
        return self._inference._pipeline

    def reconstruct(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        seed: int | None = 42,
        decode_formats: list[str] | None = None,
    ) -> ReconstructionResult:
        """
        Run SAM3D on an RGB image and binary foreground mask.

        ``decode_formats`` defaults to ``["gaussian", "mesh"]`` for MVP mesh output.
        """
        if decode_formats is None:
            decode_formats = ["gaussian", "mesh"]

        pipe = self.pipeline
        rgba = self._inference.merge_mask_to_rgba(image, mask)

        with pipe.device:
            pointmap_dict = pipe.compute_pointmap(rgba)
            pointmap = pointmap_dict["pointmap"]

            ss_input = pipe.preprocess_image(rgba, pipe.ss_preprocessor, pointmap=pointmap)
            slat_input = pipe.preprocess_image(rgba, pipe.slat_preprocessor)

            if seed is not None:
                torch.manual_seed(seed)

            ss_return = pipe.sample_sparse_structure(ss_input, inference_steps=None)

            pointmap_scale = ss_input.get("pointmap_scale")
            pointmap_shift = ss_input.get("pointmap_shift")
            ss_return.update(
                pipe.pose_decoder(
                    ss_return,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )
            ss_return["scale"] = ss_return["scale"] * ss_return["downsample_factor"]

            shape_latent = ss_return["shape"].squeeze(0).cpu()
            coords = ss_return["coords"]

            slat = pipe.sample_slat(slat_input, coords, inference_steps=None)
            slat_coords = slat.coords[:, 1:].cpu()
            slat_feats = slat.feats.cpu()

            outputs = pipe.decode_slat(slat, decode_formats)
            outputs = pipe.postprocess_slat_output(
                outputs,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                use_vertex_color=True,
            )

            # Extract DINOv2 tokens from both condition embedders (no extra model load).
            dino_cls: torch.Tensor | None = None
            dino_patches: torch.Tensor | None = None
            ss_dino_cls: torch.Tensor | None = None
            ss_dino_patches: torch.Tensor | None = None
            embedders = getattr(pipe, "condition_embedders", {})

            slat_embedder = embedders.get("slat_condition_embedder")
            if slat_embedder is not None:
                with torch.no_grad():
                    dino_tokens = slat_embedder(**slat_input)  # (1, 1+N_patches, 768)
                dino_cls = dino_tokens[0, 0].cpu().float()      # (768,)
                dino_patches = dino_tokens[0, 1:].cpu().float() # (N_patches, 768)

            ss_embedder = embedders.get("ss_condition_embedder")
            if ss_embedder is not None:
                with torch.no_grad():
                    ss_tokens = ss_embedder(**ss_input)         # (1, 1+N_patches, 768)
                ss_dino_cls = ss_tokens[0, 0].cpu().float()
                ss_dino_patches = ss_tokens[0, 1:].cpu().float()

        global_latent = slat_feats.mean(dim=0)  # (N, 8) occupied voxels → (8,)

        return ReconstructionResult(
            shape_latent=shape_latent,
            global_latent=global_latent,
            slat_feats=slat_feats,
            slat_coords=slat_coords,
            dino_cls=dino_cls,
            dino_patches=dino_patches,
            ss_dino_cls=ss_dino_cls,
            ss_dino_patches=ss_dino_patches,
            gaussian_splat=outputs.get("gs"),
            mesh_scene=outputs.get("glb"),
        )


# ---------------------------------------------------------------------------
# Latent cache helpers — call once per object, reuse every training iteration
# ---------------------------------------------------------------------------

def save_global_latent(latent: torch.Tensor, path: str | Path) -> None:
    """Persist the (8,) global shape latent for one object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"global_latent": latent}, path)


def load_global_latent(path: str | Path) -> torch.Tensor:
    """Load a cached (8,) global shape latent."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["global_latent"]


def latent_cache_path(stem: str, cache_root: str | Path) -> Path:
    """Canonical path for an object's global latent: <cache_root>/sam3d/<stem>/global_latent.pt"""
    return Path(cache_root) / "sam3d" / stem / "global_latent.pt"


def resolve_global_latent_cache_path(
    cfg: dict[str, Any],
    *,
    sam3d_run_dir: str | Path | None = None,
    object_stem: str | None = None,
    latent_path: str | Path | None = None,
    project_root_path: Path | None = None,
) -> Path | None:
    """
    Resolve where ``global_latent.pt`` should live for the given hints (file may be absent).

    Priority:

    1. ``latent_path`` — absolute or repo-relative (via :func:`utils.config.resolve_path`).
    2. ``object_stem`` + ``paths.cache_root`` from ``cfg``.
    3. ``sam3d_run_dir`` + ``reconstruction/meta.json`` ``stem`` + ``paths.cache_root``.
    """
    root = project_root_path or project_root()
    if latent_path is not None:
        return resolve_path(latent_path, root=root)

    paths_cfg = cfg.get("paths") or {}
    cache_root = resolve_path(str(paths_cfg.get("cache_root", "data/cache")), root=root)

    if object_stem is not None:
        return latent_cache_path(object_stem, cache_root)

    if sam3d_run_dir is not None:
        from utils.io import load_json

        meta_path = Path(sam3d_run_dir).expanduser().resolve() / "reconstruction" / "meta.json"
        if not meta_path.is_file():
            return None
        meta = load_json(meta_path)
        stem = meta.get("stem")
        if not stem:
            return None
        return latent_cache_path(str(stem), cache_root)

    return None


def try_load_cached_global_latent(
    cfg: dict[str, Any],
    *,
    sam3d_run_dir: str | Path | None = None,
    object_stem: str | None = None,
    latent_path: str | Path | None = None,
    project_root_path: Path | None = None,
) -> tuple[torch.Tensor | None, Path | None]:
    """
    Load the ``(sam3d_dim,)`` cached global latent when the file exists.

    Returns ``(tensor, path)`` on success, or ``(None, path_or_none)`` when missing / not resolvable.
    """
    p = resolve_global_latent_cache_path(
        cfg,
        sam3d_run_dir=sam3d_run_dir,
        object_stem=object_stem,
        latent_path=latent_path,
        project_root_path=project_root_path,
    )
    if p is None or not p.is_file():
        return None, p
    return load_global_latent(p), p
