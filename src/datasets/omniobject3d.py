"""
OmniObject3D ingestion for the GEAL-pseudolabel pipeline.

OmniObject3D ships clean, real-scanned single objects (textured meshes, point clouds, and
multi-view Blender renders) organised as ``<root>/blender_renders/<category>/<object_id>/render/``.
Because the objects are clean and isolated, SAM3D reconstructs them well — which is the whole reason
for moving off 3DAffordSplat splat renders.

This module:

  * discovers per-object render images for the categories that overlap GEAL's 23-class taxonomy
    (via ``category_map`` in ``configs/omniobject3d.yaml``),
  * stages them into a ``dataset_dir/{images,masks}/`` layout that ``scripts/generate_sam3d.py`` consumes
    (mask from the render alpha channel when present, else full image), and
  * writes a training manifest whose rows carry ``object_class`` (a GEAL class) and ``verb`` (the
    canonical affordance) so ``scripts/generate_geal_pseudolabels.py`` can label each reconstruction.

Top-level imports are kept light; PIL/numpy load lazily inside the staging helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class OmniObjectSample:
    """One staged reference view of one OmniObject3D object."""

    sample_id: str        # unique stem, e.g. "bowl__001__v000"
    category: str         # OmniObject3D folder name (e.g. "bowl")
    object_id: str        # OmniObject3D object folder name
    geal_class: str       # mapped GEAL/3D-AffordanceNet class (e.g. "Bowl")
    affordance: str       # GEAL affordance/verb to query (e.g. "contain")
    image_path: Path      # source render image on disk


def render_images_for_object(object_dir: Path) -> list[Path]:
    """Return the render RGB images for one object, searching the known render sub-layouts."""
    candidates = [
        object_dir / "render" / "images",
        object_dir / "render" / "image",
        object_dir / "render",
        object_dir,
    ]
    for d in candidates:
        if d.is_dir():
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if imgs:
                return imgs
    return []


def discover_samples(
    renders_root: Path,
    category_map: dict[str, str],
    affordance_map: dict[str, str],
    *,
    views_per_object: int = 1,
) -> list[OmniObjectSample]:
    """
    Walk ``renders_root/<category>/<object_id>/render`` and emit samples for kept categories only.

    ``category_map`` keys are matched case-insensitively against the category folder name.
    """
    renders_root = Path(renders_root)
    if not renders_root.is_dir():
        raise FileNotFoundError(f"OmniObject3D renders root not found: {renders_root}")

    lowered = {k.lower(): v for k, v in category_map.items()}
    samples: list[OmniObjectSample] = []
    for cat_dir in sorted(p for p in renders_root.iterdir() if p.is_dir()):
        geal_class = lowered.get(cat_dir.name.lower())
        if geal_class is None:
            continue
        affordance = affordance_map.get(geal_class)
        if affordance is None:
            continue
        for obj_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            imgs = render_images_for_object(obj_dir)
            for view_i, img in enumerate(imgs[: max(views_per_object, 1)]):
                stem = f"{cat_dir.name}__{obj_dir.name}__v{view_i:03d}"
                samples.append(
                    OmniObjectSample(
                        sample_id=stem,
                        category=cat_dir.name,
                        object_id=obj_dir.name,
                        geal_class=geal_class,
                        affordance=affordance,
                        image_path=img,
                    )
                )
    return samples


def _derive_foreground_mask(image_path: Path, im) -> "np.ndarray":
    """Binary uint8 (0/255) foreground mask for one staged OmniObject3D render.

    OmniObject3D ``blender_renders`` are RGB on a **black** background (no alpha), so the old
    "alpha else full-image" rule produced an all-foreground mask — which would make SAM3D try to
    reconstruct the background slab. Priority instead:

      1. RGBA alpha channel (if the render ever has one);
      2. the sibling normal map ``render/normals/<stem>_normal.png`` — background normals are
         exactly ``(0,0,0)`` regardless of object colour, so this is robust for dark objects where
         an RGB threshold would erase the object (e.g. a black monitor/bag);
      3. luminance threshold on the (dark) background as a fallback;
      4. full image only as a last resort (and when the background is clearly not dark).

    Interior holes are filled (objects are centred, not touching the border).
    """
    import numpy as np
    from PIL import Image

    if im.mode == "RGBA":
        mask = np.asarray(im.split()[-1]) > 0
    else:
        normals_path = image_path.parent.parent / "normals" / f"{image_path.stem}_normal.png"
        if normals_path.is_file():
            n = np.asarray(Image.open(normals_path).convert("RGB")).astype(np.int32)
            mask = n.sum(-1) > 4  # background normals are (0,0,0); any real surface is > 0
        else:
            rgb = np.asarray(im.convert("RGB")).astype(np.int32)
            mask = rgb.sum(-1) > 12  # assumes a dark background (OmniObject3D renders)
            if mask.mean() > 0.97:  # background was not dark -> no reliable cutout
                mask = np.ones(rgb.shape[:2], dtype=bool)

    try:
        from scipy.ndimage import binary_fill_holes

        filled = binary_fill_holes(mask)
        if filled is not None:
            mask = filled
    except Exception:
        pass
    return mask.astype("uint8") * 255


def stage_sam3d_inputs(samples: list[OmniObjectSample], dataset_dir: Path) -> None:
    """Write ``dataset_dir/images/<stem>.png`` and ``dataset_dir/masks/<stem>.png`` for each sample.

    The mask is derived per :func:`_derive_foreground_mask` (alpha → normal map → luminance → full).
    """
    from PIL import Image

    images_dir = Path(dataset_dir) / "images"
    masks_dir = Path(dataset_dir) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        im = Image.open(s.image_path)
        mask = _derive_foreground_mask(s.image_path, im)
        Image.fromarray(mask, mode="L").save(masks_dir / f"{s.sample_id}.png")
        im.convert("RGB").save(images_dir / f"{s.sample_id}.png")


def _rel(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p.resolve())


def write_manifest(
    samples: list[OmniObjectSample],
    manifest_path: Path,
    *,
    data_root: Path,
    dataset_dir: Path,
    recon_output_dir: Path,
    split: str = "train",
) -> None:
    """Write a JSONL manifest consumable by ``DataRootDataset`` / the pseudolabel + train scripts.

    ``sam3d_reconstruction_dir`` is the path ``generate_sam3d.py`` will write to (``<recon_output>/<stem>``);
    ``vertex_pseudolabel_path`` is filled in later by ``generate_geal_pseudolabels.py``.
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
                "reference_rgb_path": _rel(data_root, dataset_dir / "images" / f"{s.sample_id}.png"),
                "mask_path": _rel(data_root, dataset_dir / "masks" / f"{s.sample_id}.png"),
                "sam3d_reconstruction_dir": _rel(data_root, recon_output_dir / s.sample_id),
                "split": split,
            }
            f.write(json.dumps(row) + "\n")
