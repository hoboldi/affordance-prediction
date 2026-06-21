"""
GEAL teacher wrapper — generate per-point affordance pseudolabels.

GEAL (``DylanOrange/geal``, CVPR 2025) is used here as a **frozen teacher**. Only its 3D branch
(``model.branch_3d.Branch3D``) is needed at inference: a PointNet++ + RoBERTa cross-modal model that
maps a normalized point cloud + an affordance question to per-point affordance probabilities in
``[0, 1]`` (the forward ends in ``sigmoid``). GEAL's Gaussian-splatting 2D branch is **training only**
and is *not* used at inference, so pseudolabeling needs **no CUDA rasterizer build** — just torch,
``transformers`` (roberta-base), and GEAL's point-model deps.

The GEAL forward contract (read from ``external/geal``):

* ``pred = model(text, xyz)`` where ``xyz`` is ``(B, 3, N)`` normalized to 0.5 radius and ``text`` is a
  sequence whose ``text[0]`` is a ``list[str]`` of length ``B``; the model uses ``sentence.split('.')[1]``
  (the affordance question after the first period). Output ``pred`` is ``(B, N)`` in ``[0, 1]``.
* Model is built from the ``model_3d`` block of ``external/geal/config/evaluation.yaml`` and the
  checkpoint state lives under ``ckpt["model"]`` (loaded with ``strict=False``).

Runtime prerequisites (deferred until disk/GPU available — see project docs):

* code clone at ``<repo>/external/geal`` (``git clone --depth 1 https://github.com/DylanOrange/geal``)
* weights from https://huggingface.co/datasets/dylanorange/geal into ``external/geal/ckpt/``
  (``piad_seen.pt`` / ``piad_unseen.pt`` / ``laso_seen.pt`` / ``laso_unseen.pt``)
* ``Affordance-Question.csv`` (tiny; from the LASO/PIAD release) for the exact training phrasings —
  optional; a template question is used as a fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# GEAL taxonomy — copied verbatim from external/geal/dataset/data_utils.py so this module imports
# without GEAL on the path. These are 3D-AffordanceNet's 23 object / 18 affordance classes.
GEAL_CLASSES = [
    "Bag", "Bed", "Bowl", "Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
    "Hat", "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave", "Mug",
    "Refrigerator", "Chair", "Scissors", "Table", "TrashCan", "Vase", "Bottle",
]

GEAL_AFFORDANCES = [
    "lay", "sit", "support", "grasp", "lift", "contain", "open", "wrap_grasp", "pour",
    "move", "display", "push", "pull", "listen", "wear", "press", "cut", "stab",
]


def default_geal_root() -> Path:
    """``<repo>/external/geal`` (this file is ``<repo>/src/labeling/geal_infer.py``)."""
    return Path(__file__).resolve().parents[2] / "external" / "geal"


def normalize_point_cloud(pc: np.ndarray) -> np.ndarray:
    """Center and scale a point cloud to 0.5 radius — matches GEAL's ``normalize_point_cloud``."""
    pc = np.asarray(pc, dtype=np.float32)
    pc = pc - pc.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(pc, axis=1).max()) + 1e-9
    return (pc / scale) * 0.5


def affordance_question(
    object_class: str,
    affordance: str,
    *,
    question_csv: str | Path | None = None,
    column: str = "Question0",
) -> str:
    """
    Build the sentence GEAL expects: ``"<prefix>. <affordance question>"``. ``Branch3D`` discards the
    prefix and keeps only the text after the first period, so the prefix wording is irrelevant.

    Uses the canonical phrasing from ``Affordance-Question.csv`` (LASO/PIAD) when available; otherwise
    falls back to a simple template.
    """
    question: str | None = None
    if question_csv is not None and Path(question_csv).is_file():
        import pandas as pd

        df = pd.read_csv(question_csv)
        row = df.loc[
            (df["Object"] == object_class) & (df["Affordance"] == affordance), [column]
        ]
        if not row.empty:
            question = str(row.iloc[0][column])
    if question is None:
        question = f"Where is the region to {affordance.replace('_', ' ')} the {object_class.lower()}?"
    return f"This is a {object_class}. {question}"


class GealLabeler:
    """Loads a frozen GEAL ``Branch3D`` and predicts per-point affordance scores in ``[0, 1]``."""

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        config_path: str | Path | None = None,
        geal_root: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.geal_root = Path(geal_root) if geal_root is not None else default_geal_root()
        self.config_path = (
            Path(config_path) if config_path is not None
            else self.geal_root / "config" / "evaluation.yaml"
        )
        self._device = device
        self._model = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.geal_root.is_dir():
            raise FileNotFoundError(
                f"GEAL repo not found at {self.geal_root}. Clone it first: "
                f"git clone --depth 1 https://github.com/DylanOrange/geal {self.geal_root}"
            )
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(
                f"GEAL checkpoint not found at {self.ckpt_path}. Download from "
                f"https://huggingface.co/datasets/dylanorange/geal into {self.ckpt_path.parent}/."
            )
        import torch
        import yaml

        if str(self.geal_root) not in sys.path:
            sys.path.insert(0, str(self.geal_root))
        from model.branch_3d import Branch3D  # noqa: E402 — needs geal_root on sys.path

        cfg = yaml.safe_load(self.config_path.read_text())["model_3d"]
        cfg["training"] = False
        model = Branch3D(cfg)
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        status = model.load_state_dict(state, strict=False)
        if status.missing_keys:
            print(f"[geal] missing keys: {status.missing_keys}")
        if status.unexpected_keys:
            print(f"[geal] unexpected keys: {status.unexpected_keys}")
        self._model = model.eval().to(self.device)

    def predict(
        self,
        points_xyz: np.ndarray,
        object_class: str,
        affordance: str,
        *,
        question: str | None = None,
        question_csv: str | Path | None = None,
    ) -> np.ndarray:
        """
        Per-point affordance scores for ``points_xyz`` ``(N, 3)``.

        Returns ``(N,)`` float32 in ``[0, 1]``, aligned to the input point order.
        """
        if object_class not in GEAL_CLASSES:
            raise ValueError(
                f"object_class {object_class!r} not in GEAL taxonomy {GEAL_CLASSES}"
            )
        if affordance not in GEAL_AFFORDANCES:
            raise ValueError(
                f"affordance {affordance!r} not in GEAL taxonomy {GEAL_AFFORDANCES}"
            )
        self._ensure_model()
        import torch

        pc = normalize_point_cloud(points_xyz)                       # (N, 3)
        pt = torch.from_numpy(pc.T).float().unsqueeze(0).to(self.device)  # (1, 3, N)
        sentence = question or affordance_question(
            object_class, affordance, question_csv=question_csv
        )
        text = [[sentence]]  # text[0] == [sentence]; Branch3D keeps split('.')[1]
        with torch.no_grad():
            pred = self._model(text, pt)  # (1, N) in [0, 1]
        return pred.squeeze(0).float().cpu().numpy().astype(np.float32)
