"""
Compute the theoretical minimum (Bayes-optimal) BCE loss for each overfitting subset.

The training loss is masked BCE over visible vertices with soft pseudolabels.
If the model perfectly memorises each sample the best achievable per-vertex loss is
the binary entropy H(y) = -y·log(y) - (1-y)·log(1-y), which is > 0 for soft labels.

Usage:
    python scripts/theoretical_min_loss.py [--data-root data] [--seed 42]
"""

import argparse
import random
from pathlib import Path

import torch

from affordance.data.dataset import AffordanceDataset


def binary_entropy(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp(1e-7, 1 - 1e-7)
    return -y * torch.log(y) - (1 - y) * torch.log(1 - y)


def min_loss_for_dataset(ds: AffordanceDataset) -> dict:
    total_loss = 0.0
    total_visible = 0
    pos_fracs = []

    for i in range(len(ds)):
        sample = ds[i]
        targets: torch.Tensor = sample["pseudolabels"]          # (V,)
        visible: torch.Tensor = sample["vsem_visible"].float()  # (V,)

        n_vis = int(visible.sum().item())
        if n_vis == 0:
            continue

        entropy = binary_entropy(targets)
        sample_loss = (entropy * visible).sum().item() / n_vis
        total_loss += sample_loss
        total_visible += n_vis

        pos_frac = (targets * visible).sum().item() / n_vis
        pos_fracs.append(pos_frac)

    n = len(ds)
    return {
        "samples": n,
        "min_loss": total_loss / n,
        "avg_pos_frac": sum(pos_fracs) / len(pos_fracs) if pos_fracs else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data_root)
    split = "train"

    subsets = {
        "single  (1)": AffordanceDataset.single(root, split),
        "nine    (9)": AffordanceDataset.nine(root, split),
        "ninety (90)": AffordanceDataset.ninety(root, split),
        "full       ": AffordanceDataset(root, split),
    }

    print(f"\n{'Subset':<18}  {'N':>6}  {'Min BCE loss':>14}  {'Avg pos frac':>14}")
    print("-" * 60)
    for name, ds in subsets.items():
        result = min_loss_for_dataset(ds)
        print(
            f"{name:<18}  {result['samples']:>6}  "
            f"{result['min_loss']:>14.6f}  "
            f"{result['avg_pos_frac']:>14.4f}"
        )
    print()


if __name__ == "__main__":
    main()
