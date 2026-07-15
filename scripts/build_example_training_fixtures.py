#!/usr/bin/env python3
"""
Write ``examples/data_manifest/features/tiny_vertex_semantic.pt`` and
``examples/data_manifest/cache/sam3d/tiny/global_latent.pt`` for notebook 07 / CI.

Run from repo root (requires torch + numpy)::

    python scripts/build_example_training_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    import numpy as np
    import torch

    ex = _REPO / "examples" / "data_manifest"
    feat_dir = ex / "features"
    lat_dir = ex / "cache" / "sam3d" / "tiny"
    feat_dir.mkdir(parents=True, exist_ok=True)
    lat_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    torch.save(
        {
            "features": torch.randn(3, 512) * 0.02,
            "view_counts": torch.ones(3),
            # torch tensor, not numpy: the loader reads this bundle with weights_only=True,
            # which rejects numpy globals on torch>=2.6 (real pipeline data stores tensors too).
            "visible_in_any_view": torch.ones(3, dtype=torch.bool),
        },
        feat_dir / "tiny_vertex_semantic.pt",
    )
    torch.save({"global_latent": torch.arange(8, dtype=torch.float32) * 0.05}, lat_dir / "global_latent.pt")
    print("Wrote:", feat_dir / "tiny_vertex_semantic.pt")
    print("Wrote:", lat_dir / "global_latent.pt")


if __name__ == "__main__":
    main()
