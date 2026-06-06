from __future__ import annotations

from pathlib import Path

import pytest

from datasets.affordsplat_local_dataset import (
    AffordSplatLocalDataset,
    iter_affordsplat_local_rows,
    load_affordsplat_local_rows,
    peek_first_affordsplat_row,
    resolve_affordsplat_root,
    sample_random_affordsplat_row,
)


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_iter_rows_with_affordance_annos(tmp_path: Path) -> None:
    root = tmp_path / "AffordSplat"
    ply = root / "Seen" / "train" / "bag" / "Gaussian" / "GS_0017.ply"
    anno = root / "Seen" / "train" / "bag" / "contain" / "GS_anno_0017.ply"
    _touch(ply)
    _touch(anno)

    rows = load_affordsplat_local_rows(root, subset="Seen", split="train")
    assert len(rows) == 1
    assert rows[0].verb == "contain"
    assert rows[0].splat_path == ply.resolve()
    assert rows[0].affordance_gs_anno_path == anno.resolve()
    assert rows[0].category == "bag"


def test_iter_rows_multiple_verbs(tmp_path: Path) -> None:
    root = tmp_path / "mirror"
    ply = root / "Seen" / "val" / "bed" / "Gaussian" / "GS_0009.ply"
    a1 = root / "Seen" / "val" / "bed" / "grasp" / "GS_anno_0009.ply"
    a2 = root / "Seen" / "val" / "bed" / "sit on" / "GS_anno_0009.ply"
    _touch(ply)
    _touch(a1)
    _touch(a2)

    rows = load_affordsplat_local_rows(root, subset="Seen", split="val")
    assert len(rows) == 2
    verbs = {r.verb for r in rows}
    assert verbs == {"grasp", "sit on"}


def test_conditioning_verb_collapses_to_one_row_prefers_matching_folder(tmp_path: Path) -> None:
    root = tmp_path / "m"
    ply = root / "Seen" / "val" / "bed" / "Gaussian" / "GS_0009.ply"
    a1 = root / "Seen" / "val" / "bed" / "grasp" / "GS_anno_0009.ply"
    a2 = root / "Seen" / "val" / "bed" / "sit on" / "GS_anno_0009.ply"
    _touch(ply)
    _touch(a1)
    _touch(a2)

    rows = load_affordsplat_local_rows(
        root, subset="Seen", split="val", conditioning_verb="grasp"
    )
    assert len(rows) == 1
    assert rows[0].verb == "grasp"
    assert rows[0].affordance_gs_anno_path == a1.resolve()
    assert rows[0].extras.get("affordsplat_label_folder") == "grasp"


def test_conditioning_verb_fallback_first_sorted_folder(tmp_path: Path) -> None:
    root = tmp_path / "m"
    ply = root / "Seen" / "val" / "bed" / "Gaussian" / "GS_0009.ply"
    a1 = root / "Seen" / "val" / "bed" / "contain" / "GS_anno_0009.ply"
    a2 = root / "Seen" / "val" / "bed" / "lift" / "GS_anno_0009.ply"
    _touch(ply)
    _touch(a1)
    _touch(a2)

    rows = load_affordsplat_local_rows(
        root, subset="Seen", split="val", conditioning_verb="grasp"
    )
    assert len(rows) == 1
    assert rows[0].verb == "grasp"
    # "contain" < "lift" lexicographically → first sorted folder
    assert rows[0].affordance_gs_anno_path == a1.resolve()
    assert rows[0].extras.get("affordsplat_label_folder") == "contain"


def test_iter_rows_no_annos_defaults_verb(tmp_path: Path) -> None:
    root = tmp_path / "m"
    ply = root / "Seen" / "test" / "chair" / "Gaussian" / "GS_0001.ply"
    _touch(ply)
    rows = load_affordsplat_local_rows(root, subset="Seen", split="test")
    assert len(rows) == 1
    assert rows[0].verb == "grasp"
    assert rows[0].affordance_gs_anno_path is None


def test_category_filter(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _touch(root / "Seen" / "train" / "bag" / "Gaussian" / "GS_0001.ply")
    _touch(root / "Seen" / "train" / "bed" / "Gaussian" / "GS_0002.ply")
    rows = load_affordsplat_local_rows(root, subset="Seen", split="train", categories={"bed"})
    assert len(rows) == 1
    assert rows[0].category == "bed"


def test_dataset_getitem_keys(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _touch(root / "Seen" / "train" / "bag" / "Gaussian" / "GS_0001.ply")
    ds = AffordSplatLocalDataset(affordsplat_root=root, subset="Seen", split="train", cfg={})
    batch = ds[0]
    assert batch["sample_id"].startswith("Seen/train/bag/GS_0001")
    assert batch["splat_path"].name == "GS_0001.ply"
    assert batch["mesh"] is None
    assert batch["verb"] == "grasp"


def test_missing_subset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        next(iter_affordsplat_local_rows(tmp_path, subset="Seen"))


def test_resolve_affordsplat_root_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AFFORDANCE_AFFORDSPLAT_ROOT", str(tmp_path))
    assert resolve_affordsplat_root({}) == tmp_path.resolve()


def test_resolve_affordsplat_root_data_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "Seen").mkdir()
    monkeypatch.delenv("AFFORDANCE_AFFORDSPLAT_ROOT", raising=False)
    monkeypatch.setenv("AFFORDANCE_DATA_ROOT", str(tmp_path))
    assert resolve_affordsplat_root({}) == tmp_path.resolve()


def test_sample_random_affordsplat_row_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _touch(root / "Seen" / "train" / "bag" / "Gaussian" / "GS_0001.ply")
    _touch(root / "Seen" / "train" / "bed" / "Gaussian" / "GS_0002.ply")
    a = sample_random_affordsplat_row(root, subset="Seen", split="train", seed=12345)
    b = sample_random_affordsplat_row(root, subset="Seen", split="train", seed=12345)
    assert a is not None and b is not None
    assert a.sample_id == b.sample_id


def test_shuffle_rows_ok(tmp_path: Path) -> None:
    root = tmp_path / "m"
    for cat in ("a", "b", "c"):
        _touch(root / "Seen" / "train" / cat / "Gaussian" / "GS_0001.ply")
    ds = AffordSplatLocalDataset(
        affordsplat_root=root, subset="Seen", split="train", cfg={}, shuffle_rows=True, shuffle_seed=0
    )
    assert len(ds) == 3
    root = tmp_path / "m"
    _touch(root / "Seen" / "train" / "aaa" / "Gaussian" / "GS_0001.ply")
    _touch(root / "Seen" / "train" / "zzz" / "Gaussian" / "GS_9999.ply")
    row = peek_first_affordsplat_row(root, subset="Seen", split="train")
    assert row is not None
    assert row.category == "aaa"
    assert row.splat_path.name == "GS_0001.ply"
