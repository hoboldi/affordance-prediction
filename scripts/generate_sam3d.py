"""
Batch SAM3D reconstruction for a folder of RGB images.

Expected dataset layout (with masks):
    dataset_dir/images/, dataset_dir/masks/

With --no_masks:
    dataset_dir/images/ only (full-image foreground mask per image)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Allow running as ``python scripts/generate_sam3d.py`` without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from reconstruction.mesh_utils import ensure_decode_formats, reconstruction_paths, save_reconstruction
from reconstruction.sam3d_wrapper import SAM3DWrapper
from utils.config import load_config, resolve_path
from utils.io import (
    find_mask_file,
    full_image_mask,
    list_images,
    load_binary_mask,
    load_rgb_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM3D batch reconstruction")
    p.add_argument("--dataset_dir", required=True, help="Dataset root (images/ and optional masks/)")
    p.add_argument("--output_dir", required=True, help="Per-image reconstruction output directory")
    p.add_argument(
        "--config",
        default=None,
        help="SAM3D pipeline.yaml (default: from configs/default.yaml)",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: from config)")
    p.add_argument(
        "--decode_formats",
        nargs="+",
        default=None,
        choices=["gaussian", "mesh", "gaussian_4"],
        help="3D decode formats (default: from config)",
    )
    p.add_argument("--no_masks", action="store_true", help="Use full-image mask for every image")
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples that already have meta.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found under {dataset_dir}")
    if not args.no_masks and not masks_dir.exists():
        raise FileNotFoundError(f"masks/ not found under {dataset_dir}. Use --no_masks to skip.")

    image_paths = list_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    recon_cfg = cfg.get("reconstruction", {})
    config_path = args.config or recon_cfg.get(
        "sam3d_config", "sam-3d-objects/checkpoints/hf/pipeline.yaml"
    )
    seed = args.seed if args.seed is not None else cfg.get("project", {}).get("seed", 42)
    decode_formats = args.decode_formats or recon_cfg.get("decode_formats", ["gaussian", "mesh"])
    decode_formats = ensure_decode_formats(decode_formats)

    logger.info(f"Found {len(image_paths)} image(s); decode_formats={decode_formats}")
    wrapper = SAM3DWrapper(
        config_path,
        compile_model=recon_cfg.get("compile", False),
    )
    logger.info(f"SAM3D loaded from {resolve_path(config_path)}")

    errors: list[tuple[str, str]] = []
    for img_path in tqdm(image_paths, desc="SAM3D reconstruction"):
        stem = img_path.stem
        out_dir = output_dir / stem

        if args.skip_existing and reconstruction_paths(out_dir)["meta"].exists():
            logger.info(f"[{stem}] Skipping (already done)")
            continue

        mask_path: Path | None = None
        if not args.no_masks:
            mask_path = find_mask_file(masks_dir, stem)
            if mask_path is None:
                logger.warning(f"[{stem}] No mask found — skipping")
                errors.append((stem, "mask not found"))
                continue

        try:
            image = load_rgb_image(img_path)
            mask = full_image_mask(image) if mask_path is None else load_binary_mask(mask_path)

            result = wrapper.reconstruct(
                image,
                mask,
                seed=seed,
                decode_formats=decode_formats,
            )
            save_reconstruction(
                result,
                out_dir,
                stem=stem,
                seed=seed,
                image_path=img_path,
                mask_path=mask_path,
            )
            logger.info(f"[{stem}] Done — SLAT voxels: {result.slat_coords.shape[0]}")
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"[{stem}] Failed:\n{tb}")
            errors.append((stem, tb))

    total = len(image_paths)
    failed = len(errors)
    logger.info(f"Finished: {total - failed}/{total} succeeded, {failed} failed.")
    for stem, reason in errors:
        logger.warning(f"  {stem}: {reason[:120]}")


if __name__ == "__main__":
    main()
