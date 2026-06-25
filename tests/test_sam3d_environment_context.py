"""``sam3d_environment`` restores ``sys.path`` and ``LIDRA_SKIP_INIT`` (see ``sam3d_wrapper``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from reconstruction.sam3d_wrapper import sam3d_environment
from utils.config import project_root


def test_sam3d_environment_restores_sys_path_and_lidra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "notebook").mkdir(parents=True)
    (tmp_path / "notebook" / "inference.py").write_text("# stub\n", encoding="utf-8")
    monkeypatch.setenv("SAM3D_OBJECTS_ROOT", str(tmp_path))

    n_before = len(sys.path)
    lidra_key = "LIDRA_SKIP_INIT"
    monkeypatch.delenv(lidra_key, raising=False)

    with sam3d_environment(project_root()):
        assert str(tmp_path) in sys.path
        assert str(tmp_path / "notebook") in sys.path
        assert os.environ.get(lidra_key) == "true"

    assert len(sys.path) == n_before
    assert str(tmp_path) not in sys.path
    assert str(tmp_path / "notebook") not in sys.path
    assert lidra_key not in os.environ
