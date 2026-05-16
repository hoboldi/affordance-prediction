"""
Dataset pipeline: for each image in a dataset, generate a 3D model (gaussian splat + mesh)
and save both latent representations (shape latent from stage 1, SLAT from stage 2).

Expected dataset layout (with masks):
    dataset_dir/
        images/   ← RGB images (.jpg, .jpeg, .png, .webp)
        masks/    ← binary masks with the same stem as the images (.png)

Expected dataset layout (--no_masks):
    dataset_dir/
        images/   ← RGB images (.jpg, .jpeg, .png, .webp)
    A full-image mask (all pixels = foreground) is used for every image.

Output layout:
    output_dir/
        {image_stem}/
            shape_latent.pt   ← (4096, 8) float tensor, stage-1 global shape latent
            slat_feats.pt     ← (N, 8) float tensor, stage-2 per-voxel features
            slat_coords.pt    ← (N, 3) int tensor, stage-2 voxel coordinates
            gaussian.ply      ← Gaussian splat
            mesh.glb          ← Textured mesh
            meta.json         ← seed, source paths, slat voxel count
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path

# Allow importing from the submodule without installing it
sys.path.insert(0, str(Path(__file__).parent / "sam-3d-objects"))
sys.path.insert(0, str(Path(__file__).parent / "sam-3d-objects" / "notebook"))

os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
os.environ["LIDRA_SKIP_INIT"] = "true"

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from loguru import logger

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_mask(masks_dir: Path, stem: str) -> Path | None:
    for ext in MASK_EXTENSIONS:
        p = masks_dir / (stem + ext)
        if p.exists():
            return p
    return None


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    m = Image.open(path)
    arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[..., -1]
    return (arr > 0).astype(bool)


def full_image_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape[:2], dtype=bool)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_single(
    inference,
    image: np.ndarray,
    mask: np.ndarray,
    seed: int,
    decode_formats: list[str],
) -> dict:
    """
    Run the full SAM3D pipeline for one image and return all outputs.

    Returns a dict with keys:
        shape_latent  (torch.Tensor, cpu)
        slat_feats    (torch.Tensor, cpu)
        slat_coords   (torch.Tensor, cpu)
        gs            gaussian splat object (or None)
        glb           trimesh scene (or None)
    """
    pipe = inference._pipeline

    rgba = inference.merge_mask_to_rgba(image, mask)

    with pipe.device:
        pointmap_dict = pipe.compute_pointmap(rgba)
        pointmap = pointmap_dict["pointmap"]

        ss_input = pipe.preprocess_image(rgba, pipe.ss_preprocessor, pointmap=pointmap)
        slat_input = pipe.preprocess_image(rgba, pipe.slat_preprocessor)

        if seed is not None:
            torch.manual_seed(seed)

        # Stage 1 – sparse structure → shape latent
        ss_return = pipe.sample_sparse_structure(ss_input, inference_steps=None)

        pointmap_scale = ss_input.get("pointmap_scale", None)
        pointmap_shift = ss_input.get("pointmap_shift", None)
        ss_return.update(
            pipe.pose_decoder(ss_return, scene_scale=pointmap_scale, scene_shift=pointmap_shift)
        )
        ss_return["scale"] = ss_return["scale"] * ss_return["downsample_factor"]

        shape_latent = ss_return["shape"].squeeze(0).cpu()  # (4096, 8)
        coords = ss_return["coords"]

        # Stage 2 – structured latent (SLAT)
        slat = pipe.sample_slat(slat_input, coords, inference_steps=None)

        # SparseTensor: coords (N,4) [batch, x, y, z], feats (N, 8)
        slat_coords = slat.coords[:, 1:].cpu()  # (N, 3)
        slat_feats = slat.feats.cpu()           # (N, 8)

        # Decode to 3D representations
        outputs = pipe.decode_slat(slat, decode_formats)
        outputs = pipe.postprocess_slat_output(
            outputs,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
        )

    return {
        "shape_latent": shape_latent,
        "slat_coords": slat_coords,
        "slat_feats": slat_feats,
        "gs": outputs.get("gs"),
        "glb": outputs.get("glb"),
    }


def save_results(results: dict, out_dir: Path, stem: str, seed: int, image_path: Path, mask_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(results["shape_latent"], out_dir / "shape_latent.pt")
    torch.save(results["slat_feats"], out_dir / "slat_feats.pt")
    torch.save(results["slat_coords"], out_dir / "slat_coords.pt")

    if results["gs"] is not None:
        results["gs"].save_ply(str(out_dir / "gaussian.ply"))

    if results["glb"] is not None:
        results["glb"].export(str(out_dir / "mesh.glb"))

    meta = {
        "stem": stem,
        "seed": seed,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "shape_latent_shape": list(results["shape_latent"].shape),
        "slat_voxel_count": int(results["slat_coords"].shape[0]),
        "slat_feats_shape": list(results["slat_feats"].shape),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAM3D dataset pipeline")
    p.add_argument("--dataset_dir", required=True, help="Root of the dataset (must contain images/ and masks/)")
    p.add_argument("--output_dir", required=True, help="Where to write per-image outputs")
    p.add_argument(
        "--config",
        default="sam-3d-objects/checkpoints/hf/pipeline.yaml",
        help="Path to the SAM3D pipeline.yaml config (default: sam-3d-objects/checkpoints/hf/pipeline.yaml)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--decode_formats",
        nargs="+",
        default=["gaussian", "mesh"],
        choices=["gaussian", "mesh", "gaussian_4"],
        help="3D decode formats to generate (default: gaussian mesh)",
    )
    p.add_argument(
        "--no_masks",
        action="store_true",
        help="Skip mask loading and use a full-image mask for every image",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images whose output directory already contains meta.json",
    )
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"images/ directory not found under {dataset_dir}")
    if not args.no_masks and not masks_dir.exists():
        raise FileNotFoundError(
            f"masks/ directory not found under {dataset_dir}. "
            "Use --no_masks to run without masks."
        )

    # Collect image files
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    logger.info(f"Found {len(image_paths)} image(s) in {images_dir}")

    # postprocess_slat_output always reads outputs["gaussian"] when "mesh" is present
    decode_formats = args.decode_formats
    if "mesh" in decode_formats and "gaussian" not in decode_formats:
        logger.warning("Adding 'gaussian' to decode_formats because mesh decoding requires it")
        decode_formats = ["gaussian"] + decode_formats

    # Load model once
    from inference import Inference  # noqa: E402 – needs sys.path set up above

    logger.info(f"Loading SAM3D pipeline from {args.config} ...")
    inference = Inference(args.config, compile=False)
    logger.info("Pipeline loaded.")

    errors = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        stem = img_path.stem
        out_dir = output_dir / stem

        if args.skip_existing and (out_dir / "meta.json").exists():
            logger.info(f"[{stem}] Skipping (already done)")
            continue

        if args.no_masks:
            mask_path = None
        else:
            mask_path = find_mask(masks_dir, stem)
            if mask_path is None:
                logger.warning(f"[{stem}] No mask found in {masks_dir} – skipping")
                errors.append((stem, "mask not found"))
                continue

        try:
            image = load_rgb(img_path)
            mask = full_image_mask(image) if mask_path is None else load_mask(mask_path)

            results = process_single(
                inference, image, mask, seed=args.seed, decode_formats=decode_formats
            )
            save_results(results, out_dir, stem, args.seed, img_path, mask_path)
            logger.info(
                f"[{stem}] Done — SLAT voxels: {results['slat_coords'].shape[0]}"
            )
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"[{stem}] Failed:\n{tb}")
            errors.append((stem, tb))

    # Summary
    total = len(image_paths)
    failed = len(errors)
    logger.info(f"Finished: {total - failed}/{total} succeeded, {failed} failed.")
    if errors:
        logger.warning("Failed items:")
        for stem, reason in errors:
            logger.warning(f"  {stem}: {reason[:120]}")


if __name__ == "__main__":
    main()
