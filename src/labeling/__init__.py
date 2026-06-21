"""Pseudolabel generation from frozen teacher models (currently GEAL)."""

from labeling.geal_infer import (
    GEAL_AFFORDANCES,
    GEAL_CLASSES,
    GealLabeler,
    affordance_question,
    default_geal_root,
)

__all__ = [
    "GEAL_AFFORDANCES",
    "GEAL_CLASSES",
    "GealLabeler",
    "affordance_question",
    "default_geal_root",
]
