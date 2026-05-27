"""Tests for view-selection winner tie-break (``gsplat_viewpoint_selection``)."""

from __future__ import annotations

from rendering.gsplat_viewpoint_selection import _pick_best_view_index


def test_pick_best_prefers_lower_chamfer_on_score_tie() -> None:
    scores = [1.0, 1.0, 1.0]
    chamfers = [0.5, 0.2, 0.4]
    assert _pick_best_view_index(scores, chamfers) == 1


def test_pick_best_argmax_when_scores_distinct() -> None:
    scores = [0.1, 0.9, 0.3]
    chamfers = [1.0, 0.0, 0.5]
    assert _pick_best_view_index(scores, chamfers) == 1


def test_pick_best_lower_index_when_both_tied() -> None:
    scores = [2.0, 2.0]
    chamfers = [0.1, 0.1]
    assert _pick_best_view_index(scores, chamfers) == 0
