import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

VERBS = ["contain", "display", "grasp", "lift", "move", "open", "pour", "press", "sit"]
VERB_TO_IDX = {v: i for i, v in enumerate(VERBS)}

_RECONSTRUCTION_FILES = [
    "vertex_positions",
    "vertex_normals",
    "slat_coords",
    "slat_feats",
    "slat_vertex_features",
    "dino_cls",
    "ss_dino_cls",
    "dino_patches",
    "shape_latent",
]


class AffordanceDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str = "train", samples: list[dict] | None = None):
        self.data_root = Path(data_root)
        if samples is not None:
            self.samples = samples
        else:
            manifest_path = self.data_root / "manifest.pseudolabeled.vsem.jsonl"
            with open(manifest_path) as f:
                self.samples = [
                    json.loads(line)
                    for line in f
                    if json.loads(line)["split"] == split
                ]

    @classmethod
    def _load_manifest(cls, data_root: Path, split: str) -> list[dict]:
        manifest_path = data_root / "manifest.pseudolabeled.vsem.jsonl"
        with open(manifest_path) as f:
            return [json.loads(line) for line in f if json.loads(line)["split"] == split]

    @classmethod
    def single(cls, data_root: str | Path, split: str = "train") -> "AffordanceDataset":
        all_samples = cls._load_manifest(Path(data_root), split)
        return cls(data_root, split, samples=[all_samples[0]])

    @classmethod
    def balanced(cls, data_root: str | Path, split: str = "train", per_verb: int = 1) -> "AffordanceDataset":
        """Sample `per_verb` examples per verb, ensuring equal verb distribution."""
        all_samples = cls._load_manifest(Path(data_root), split)
        by_verb: dict[str, list[dict]] = {v: [] for v in VERBS}
        for s in all_samples:
            if s["verb"] in by_verb:
                by_verb[s["verb"]].append(s)
        selected = []
        for verb_samples in by_verb.values():
            selected.extend(random.sample(verb_samples, min(per_verb, len(verb_samples))))
        return cls(data_root, split, samples=selected)

    @classmethod
    def nine(cls, data_root: str | Path, split: str = "train") -> "AffordanceDataset":
        return cls.balanced(data_root, split, per_verb=1)

    @classmethod
    def ninety(cls, data_root: str | Path, split: str = "train") -> "AffordanceDataset":
        return cls.balanced(data_root, split, per_verb=10)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        entry = self.samples[idx]
        rec_dir = self.data_root / entry["sam3d_reconstruction_dir"]

        sample = {
            "sample_id": entry["sample_id"],
            "verb": entry["verb"],
            "verb_idx": torch.tensor(VERB_TO_IDX[entry["verb"]], dtype=torch.long),
            "category": entry["category"],
        }

        for name in _RECONSTRUCTION_FILES:
            sample[name] = torch.load(rec_dir / f"{name}.pt", map_location="cpu", weights_only=True)

        sample["vertex_affordance"] = torch.load(
            self.data_root / entry["vertex_pseudolabel_path"],
            map_location="cpu",
            weights_only=True,
        )
        vsem = torch.load(
            self.data_root / entry["vertex_semantics_path"],
            map_location="cpu",
            weights_only=False,
        )
        sample["vertex_features"] = vsem["features"]
        sample["vertex_visible_mask"] = vsem["visible_in_any_view"]

        return sample
