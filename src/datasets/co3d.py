"""
CO3D v2 ingestion for the GEAL-pseudolabel pipeline (REAL images).

CO3D (Common Objects in 3D, Meta) provides real cellphone-captured turntable videos of single objects,
each frame paired with a foreground mask. Unlike OmniObject3D's synthetic ``blender_renders``, these are
real photographs — SAM3D's native input domain. We use ONE RGB frame + its provided mask per sequence
(object); CO3D's depth maps / point clouds are ignored (SAM3D estimates its own geometry).

Layout (CO3D v2)::

    <co3d_root>/<category>/<sequence_name>/images/frame<NNNNNN>.jpg
    <co3d_root>/<category>/<sequence_name>/masks/frame<NNNNNN>.png   # foreground mask (uint8)
    <co3d_root>/<category>/{frame_annotations.jgz, sequence_annotations.jgz, set_lists/, eval_batches/}

This module mirrors :mod:`datasets.omniobject3d`: ``discover_samples`` -> ``stage_sam3d_inputs`` ->
``write_manifest``. Only categories in ``category_map`` (CO3D folder -> GEAL class) are kept. Categories
in ``val_categories`` are marked split ``"val"`` (held-out unseen-class eval); the rest are ``"train"``.

Top-level imports are kept light; PIL/numpy load lazily inside the staging helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# CO3D per-category entries that are metadata, not object sequences.
_NON_SEQUENCE_DIRS = {"set_lists", "eval_batches"}


@dataclass
class CO3DSample:
    """One staged reference view of one CO3D object (sequence)."""

    sample_id: str          # "<category>__<sequence>__v000"
    category: str           # CO3D folder name (e.g. "bowl")
    sequence: str           # CO3D sequence name (object id, e.g. "34_1403_4393")
    geal_class: str         # mapped GEAL/3D-AffordanceNet class (e.g. "Bowl")
    affordance: str         # GEAL affordance/verb to query (e.g. "contain")
    image_path: Path        # chosen RGB frame
    mask_path: Path | None  # CO3D foreground mask for that frame (or None if missing)
    split: str = "train"    # "train" | "val"


def _select_frames(frames: list[Path], views: int, mode: str = "middle") -> list[Path]:
    """Pick ``views`` frames from a sequence's sorted frame list.

    ``middle`` (default) returns the central frame for ``views == 1`` (a clean canonical view of a
    turntable capture); ``first`` returns the leading frames; otherwise frames are spread evenly
    across the sequence (endpoints included).
    """
    n = len(frames)
    if n == 0:
        return []
    views = max(views, 1)
    if views >= n:
        return list(frames)
    if mode == "first":
        return frames[:views]
    if views == 1:
        return [frames[n // 2]]
    return [frames[round(i * (n - 1) / (views - 1))] for i in range(views)]


def discover_samples(
    co3d_root: Path,
    category_map: dict[str, str],
    affordance_map: dict[str, str],
    *,
    views_per_object: int = 1,
    frame_selection: str = "middle",
    max_objects_per_category: int | None = None,
    val_categories: list[str] | None = None,
) -> list[CO3DSample]:
    """
    Walk ``co3d_root/<category>/<sequence>/images`` and emit samples for kept categories only.

    ``category_map`` keys are matched case-insensitively against the category folder name. Sequences
    in ``val_categories`` get split ``"val"``; ``max_objects_per_category`` caps sequences per category.
    """
    co3d_root = Path(co3d_root)
    if not co3d_root.is_dir():
        raise FileNotFoundError(f"CO3D root not found: {co3d_root}")

    lowered = {k.lower(): v for k, v in category_map.items()}
    val_set = {c.lower() for c in (val_categories or [])}
    samples: list[CO3DSample] = []

    for cat_dir in sorted(p for p in co3d_root.iterdir() if p.is_dir()):
        geal_class = lowered.get(cat_dir.name.lower())
        if geal_class is None:
            continue
        affordance = affordance_map.get(geal_class)
        if affordance is None:
            continue
        split = "val" if cat_dir.name.lower() in val_set else "train"

        seq_dirs = sorted(
            p for p in cat_dir.iterdir() if p.is_dir() and p.name not in _NON_SEQUENCE_DIRS
        )
        kept = 0
        for seq_dir in seq_dirs:
            if max_objects_per_category is not None and kept >= max_objects_per_category:
                break
            images_dir = seq_dir / "images"
            if not images_dir.is_dir():
                continue
            frames = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not frames:
                continue
            masks_dir = seq_dir / "masks"
            for view_i, img in enumerate(_select_frames(frames, views_per_object, frame_selection)):
                mask = masks_dir / f"{img.stem}.png"
                samples.append(
                    CO3DSample(
                        sample_id=f"{cat_dir.name}__{seq_dir.name}__v{view_i:03d}",
                        category=cat_dir.name,
                        sequence=seq_dir.name,
                        geal_class=geal_class,
                        affordance=affordance,
                        image_path=img,
                        mask_path=mask if mask.is_file() else None,
                        split=split,
                    )
                )
            kept += 1
    return samples


def stage_sam3d_inputs(samples: list[CO3DSample], dataset_dir: Path) -> None:
    """Write ``dataset_dir/images/<stem>.png`` + ``dataset_dir/masks/<stem>.png`` for each sample.

    CO3D ships a foreground mask per frame, so the mask is the binarised CO3D mask (threshold at the
    0.5 midpoint); when a mask is missing or empty we fall back to a full-image mask.
    """
    import numpy as np
    from PIL import Image

    images_dir = Path(dataset_dir) / "images"
    masks_dir = Path(dataset_dir) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        im = Image.open(s.image_path).convert("RGB")
        if s.mask_path is not None and s.mask_path.is_file():
            m = np.asarray(Image.open(s.mask_path).convert("L"))
            mask = (m > 127).astype("uint8") * 255
            if mask.mean() < 1.0:  # essentially empty -> mask unusable
                mask = np.full(m.shape, 255, dtype="uint8")
        else:
            w, h = im.size
            mask = np.full((h, w), 255, dtype="uint8")
        Image.fromarray(mask, mode="L").save(masks_dir / f"{s.sample_id}.png")
        im.save(images_dir / f"{s.sample_id}.png")


def _rel(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p.resolve())


def write_manifest(
    samples: list[CO3DSample],
    manifest_path: Path,
    *,
    data_root: Path,
    dataset_dir: Path,
    recon_output_dir: Path,
) -> None:
    """Write a JSONL manifest consumable by ``DataRootDataset`` / the pseudolabel + train scripts.

    ``sam3d_reconstruction_dir`` is the path ``generate_sam3d.py`` will write to (``<recon_output>/<stem>``);
    ``vertex_pseudolabel_path`` is filled in later by ``generate_geal_pseudolabels.py``. ``split`` is per
    sample (``"train"`` / ``"val"``), driven by ``val_categories`` in :func:`discover_samples`.
    """
    data_root = Path(data_root)
    dataset_dir = Path(dataset_dir)
    recon_output_dir = Path(recon_output_dir)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        for s in samples:
            row = {
                "sample_id": s.sample_id,
                "verb": s.affordance,
                "object_class": s.geal_class,
                "category": s.category,
                "sequence": s.sequence,
                "reference_rgb_path": _rel(data_root, dataset_dir / "images" / f"{s.sample_id}.png"),
                "mask_path": _rel(data_root, dataset_dir / "masks" / f"{s.sample_id}.png"),
                "sam3d_reconstruction_dir": _rel(data_root, recon_output_dir / s.sample_id),
                "split": s.split,
            }
            f.write(json.dumps(row) + "\n")
