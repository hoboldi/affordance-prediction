from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.config import project_root, resolve_path


@dataclass
class ReconstructionResult:
    """Outputs from a single SAM3D forward pass."""

    shape_latent: torch.Tensor  # (4096, 8)
    slat_feats: torch.Tensor  # (N, 8)
    slat_coords: torch.Tensor  # (N, 3)
    gaussian_splat: Any | None = None
    mesh_scene: Any | None = None


def configure_sam3d_environment(root: Path | None = None) -> Path:
    """
    Add sam-3d-objects to ``sys.path`` and set env vars required by the submodule.

    Returns the sam-3d-objects directory path.
    """
    root = root or project_root()
    sam3d_root = root / "sam-3d-objects"
    notebook_dir = sam3d_root / "notebook"

    for path in (sam3d_root, notebook_dir):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
    os.environ["LIDRA_SKIP_INIT"] = "true"
    return sam3d_root


class SAM3DWrapper:
    """Thin wrapper around the SAM3D ``Inference`` class."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        compile_model: bool = False,
        project_root_path: Path | None = None,
    ) -> None:
        root = project_root_path or project_root()
        configure_sam3d_environment(root)

        from inference import Inference  # noqa: E402

        resolved = resolve_path(config_path, root=root)
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

        return ReconstructionResult(
            shape_latent=shape_latent,
            slat_feats=slat_feats,
            slat_coords=slat_coords,
            gaussian_splat=outputs.get("gs"),
            mesh_scene=outputs.get("glb"),
        )
