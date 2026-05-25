"""
Load **3DAffordSplat / AffordSplat** samples from a local Hugging Face checkout (no Hub download).

Expected layout (from the `Weizm/AffordSplat` dataset card)::

    <affordsplat_root>/
      Seen/
        train|val|test/
          <category>/          # e.g. bag, bed
            Gaussian/
              GS_0001.ply
            <affordance_verb>/ # e.g. grasp, contain — skip Gaussian/PointCloud
              GS_anno_0001.ply
      ...

Set ``AFFORDANCE_AFFORDSPLAT_ROOT`` to the directory that **contains** ``Seen/``, or rely on
auto-detection: ``AFFORDANCE_DATA_ROOT`` / ``paths.data_root`` when ``<that>/Seen`` exists,
then ``/workspace/data``, then ``/data`` (typical mounts).
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset

from datasets.data_root_dataset import resolve_data_root
from utils.config import load_config, resolve_path

_SKIP_CATEGORY_SUBDIRS = frozenset(
    {
        "gaussian",
        "pointcloud",
    }
)


@dataclass
class AffordSplatLocalRow:
    """One training index: Gaussian splat + optional affordance annotation ply."""

    sample_id: str
    verb: str
    subset: str
    split: str
    category: str
    gaussian_stem: str
    splat_path: Path
    affordance_gs_anno_path: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def resolve_affordsplat_root(cfg: dict[str, Any] | None = None) -> Path | None:
    """
    Root directory containing ``Seen/`` (and optionally ``UnSeen/``).

    Priority:

    1. ``AFFORDANCE_AFFORDSPLAT_ROOT`` (must be the parent of ``Seen/``).
    2. ``paths.affordsplat_root`` in YAML (repo-relative or absolute).
    3. ``AFFORDANCE_DATA_ROOT`` if ``<root>/Seen`` exists (Docker: ``/workspace/data``).
    4. ``paths.data_root`` from config (same as :func:`datasets.data_root_dataset.resolve_data_root`)
       if ``<resolved>/Seen`` exists.
    5. ``/workspace/data`` then ``/data`` when ``…/Seen`` exists.
    """
    env = os.environ.get("AFFORDANCE_AFFORDSPLAT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    cfg = cfg if cfg is not None else load_config()
    paths = cfg.get("paths") or {}
    raw = paths.get("affordsplat_root")
    if raw:
        p = resolve_path(str(raw))
        return p.expanduser().resolve()

    def _with_seen(root: Path) -> Path | None:
        r = root.expanduser().resolve()
        return r if (r / "Seen").is_dir() else None

    data_env = os.environ.get("AFFORDANCE_DATA_ROOT", "").strip()
    if data_env:
        got = _with_seen(Path(data_env))
        if got is not None:
            return got

    got = _with_seen(resolve_data_root(cfg))
    if got is not None:
        return got

    for candidate in (Path("/workspace/data"), Path("/data")):
        got = _with_seen(candidate)
        if got is not None:
            return got

    return None


_GS_PLY = re.compile(r"^GS_(\d+)\.ply$", re.IGNORECASE)


def _gaussian_numeric_id(filename: str) -> str | None:
    m = _GS_PLY.match(filename)
    return m.group(1) if m else None


def _iter_verb_anno_paths(category_dir: Path, numeric_id: str) -> dict[str, Path]:
    """
    Map affordance folder name → ``GS_anno_<id>.ply`` if that file exists.
    """
    out: dict[str, Path] = {}
    want = f"GS_anno_{numeric_id}.ply"
    for child in category_dir.iterdir():
        if not child.is_dir():
            continue
        key = child.name.lower()
        if key in _SKIP_CATEGORY_SUBDIRS:
            continue
        cand = child / want
        if cand.is_file():
            out[child.name] = cand.resolve()
    return out


def iter_affordsplat_local_rows(
    affordsplat_root: Path,
    *,
    subset: str = "Seen",
    split: str | None = None,
    categories: set[str] | None = None,
) -> Iterator[AffordSplatLocalRow]:
    root = Path(affordsplat_root).resolve()
    subset_path = root / subset
    if not subset_path.is_dir():
        raise FileNotFoundError(f"AffordSplat subset not found: {subset_path}")

    split_names = ["train", "val", "test"]
    if split is not None:
        if split not in split_names:
            raise ValueError(f"split must be one of {split_names} or None, got {split!r}")
        split_names = [split]

    for split_name in split_names:
        split_path = subset_path / split_name
        if not split_path.is_dir():
            continue
        for cat_dir in sorted(split_path.iterdir()):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            if categories is not None and category not in categories:
                continue
            gdir = cat_dir / "Gaussian"
            if not gdir.is_dir():
                continue
            for ply in sorted(gdir.glob("GS_*.ply")):
                if not ply.is_file():
                    continue
                num_id = _gaussian_numeric_id(ply.name)
                if num_id is None:
                    continue
                stem = ply.stem
                splat_path = ply.resolve()
                verb_to_anno = _iter_verb_anno_paths(cat_dir, num_id)

                base_extras: dict[str, Any] = {
                    "affordsplat_subset": subset,
                    "affordsplat_category": category,
                    "affordsplat_gaussian_numeric_id": num_id,
                }

                if verb_to_anno:
                    for verb, anno_path in sorted(verb_to_anno.items()):
                        sid = f"{subset}/{split_name}/{category}/{stem}/{verb}"
                        yield AffordSplatLocalRow(
                            sample_id=sid,
                            verb=verb,
                            subset=subset,
                            split=split_name,
                            category=category,
                            gaussian_stem=stem,
                            splat_path=splat_path,
                            affordance_gs_anno_path=anno_path,
                            extras={**base_extras, "affordsplat_gs_anno_path": str(anno_path)},
                        )
                else:
                    sid = f"{subset}/{split_name}/{category}/{stem}"
                    yield AffordSplatLocalRow(
                        sample_id=sid,
                        verb="grasp",
                        subset=subset,
                        split=split_name,
                        category=category,
                        gaussian_stem=stem,
                        splat_path=splat_path,
                        affordance_gs_anno_path=None,
                        extras=dict(base_extras),
                    )


def load_affordsplat_local_rows(
    affordsplat_root: Path | str,
    *,
    subset: str = "Seen",
    split: str | None = None,
    categories: set[str] | None = None,
) -> list[AffordSplatLocalRow]:
    return list(
        iter_affordsplat_local_rows(
            affordsplat_root,
            subset=subset,
            split=split,
            categories=categories,
        )
    )


def peek_first_affordsplat_row(
    affordsplat_root: Path | str | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    subset: str = "Seen",
    split: str | None = "train",
    categories: set[str] | None = None,
) -> AffordSplatLocalRow | None:
    """
    First Gaussian row without building the full index (cheap for notebooks / splat auto-pick).

    Returns ``None`` if the root is unset, ``Seen/`` (or ``subset``) is missing, or there are no
    ``GS_*.ply`` files under the chosen split.
    """
    if affordsplat_root is not None:
        root = Path(affordsplat_root).expanduser().resolve()
    else:
        root = resolve_affordsplat_root(cfg if cfg is not None else load_config())
    if root is None:
        return None
    subset_path = root / subset
    if not subset_path.is_dir():
        return None
    try:
        return next(
            iter_affordsplat_local_rows(
                root,
                subset=subset,
                split=split,
                categories=categories,
            )
        )
    except StopIteration:
        return None


def sample_random_affordsplat_row(
    affordsplat_root: Path | str | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    subset: str = "Seen",
    split: str | None = "train",
    categories: set[str] | None = None,
    seed: int | None = None,
) -> AffordSplatLocalRow | None:
    """
    Uniform random row over the same stream as :func:`iter_affordsplat_local_rows` **without**
    materialising the full list (reservoir sampling, one pass).

    ``seed=None`` uses non-deterministic RNG so each call can pick a different sample. Pass an
    ``int`` for reproducible draws.
    """
    if affordsplat_root is not None:
        root = Path(affordsplat_root).expanduser().resolve()
    else:
        root = resolve_affordsplat_root(cfg if cfg is not None else load_config())
    if root is None:
        return None
    subset_path = root / subset
    if not subset_path.is_dir():
        return None

    rng = random.Random(seed) if seed is not None else random.Random()
    chosen: AffordSplatLocalRow | None = None
    n = 0
    for row in iter_affordsplat_local_rows(
        root,
        subset=subset,
        split=split,
        categories=categories,
    ):
        n += 1
        if rng.randrange(n) == 0:
            chosen = row
    return chosen


class AffordSplatLocalDataset(Dataset[dict[str, Any]]):
    """
    PyTorch ``Dataset`` over on-disk AffordSplat / 3DAffordSplat Gaussians.

    Each item aligns with :class:`DataRootDataset` keys where possible so the same training
    code can consume ``splat_path``, ``verb``, ``sample_id``, and ``extras``.
    """

    def __init__(
        self,
        *,
        affordsplat_root: Path | str | None = None,
        cfg: dict[str, Any] | None = None,
        subset: str = "Seen",
        split: str | None = None,
        categories: list[str] | None = None,
        shuffle_rows: bool = False,
        shuffle_seed: int | None = None,
    ) -> None:
        cfg = cfg if cfg is not None else load_config()
        if affordsplat_root is not None:
            self._root = Path(affordsplat_root).expanduser().resolve()
        else:
            resolved = resolve_affordsplat_root(cfg)
            if resolved is None:
                raise ValueError(
                    "AffordSplat root not configured: set AFFORDANCE_AFFORDSPLAT_ROOT, "
                    "paths.affordsplat_root in YAML, pass affordsplat_root=, or unzip Seen/ under "
                    "AFFORDANCE_DATA_ROOT / /workspace/data / /data so <root>/Seen exists."
                )
            self._root = resolved

        cat_set = set(categories) if categories is not None else None
        self._rows = load_affordsplat_local_rows(
            self._root,
            subset=subset,
            split=split,
            categories=cat_set,
        )
        if not self._rows:
            raise FileNotFoundError(
                f"No AffordSplat Gaussian rows under {self._root!s} "
                f"(subset={subset!r}, split={split!r}). Check layout: …/Seen/<split>/<cat>/Gaussian/GS_*.ply"
            )

        if shuffle_rows:
            rng = random.Random(shuffle_seed) if shuffle_seed is not None else random.Random()
            rng.shuffle(self._rows)

    @property
    def affordsplat_root(self) -> Path:
        return self._root

    def __len__(self) -> int:
        return len(self._rows)

    def row(self, index: int) -> AffordSplatLocalRow:
        return self._rows[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self._rows[index]
        ex = dict(r.extras)
        return {
            "sample_id": r.sample_id,
            "verb": r.verb,
            "split": r.split,
            "splat_path": r.splat_path,
            "mesh_path": None,
            "mesh": None,
            "reference_rgb_path": None,
            "mask_path": None,
            "vertex_affordance_path": None,
            "vertex_affordance": None,
            "sam3d_global_latent_path": None,
            "extras": ex,
        }
