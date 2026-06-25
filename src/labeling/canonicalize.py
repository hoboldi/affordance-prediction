"""GEAL-guided orientation canonicalization for SAM3D reconstructions.

SAM3D outputs objects in inconsistent orientations (it does not preserve the input camera frame), and
the frozen GEAL teacher is only reliable near the upright canonical pose of its training data
(3D-AffordanceNet): empirically it is yaw-tolerant but produces wrong labels for flipped/tipped inputs
(see the rotation-sensitivity test in the project notes — flip/tip -> IoU 0 vs upright).

This module finds the rotation that makes GEAL's prediction most **yaw-stable** — a validated proxy for
"in-distribution / upright" — over a small candidate set of up-axes (world axes + the point cloud's PCA
axes). On a known-upright chair the true +Y axis wins the yaw-stability score decisively, and the search
leaves already-upright objects unchanged.

Usage (see scripts/generate_geal_pseudolabels.py): compute ``R`` once per object, rotate the sampled
points by ``R`` before calling ``GealLabeler.predict``; the per-point scores still correspond to the
original (un-rotated) points, so they map back onto the mesh vertices unchanged. ``R`` is also saved so
the training head can rotate ``vertex_positions``/``vertex_normals`` into the same canonical frame.
"""
from __future__ import annotations

import numpy as np


def _basis_to_up(u: np.ndarray) -> np.ndarray:
    """Rotation (3,3) sending direction ``u`` to canonical +Y (yaw fixed deterministically)."""
    u = np.asarray(u, np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    ref = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = ref - (ref @ u) * u
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e3 = np.cross(e1, u)
    return np.stack([e1, u, e3], 0).astype(np.float32)  # rows => (R @ u) == [0,1,0]


def _yaw(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def _candidate_ups(points: np.ndarray) -> list[np.ndarray]:
    """Candidate up-axes: the 6 world axes only.

    SAM3D outputs are roughly axis-aligned, so restricting to the world axes keeps the canonical pose
    clean and axis-aligned. Diagonal PCA-axis candidates were dropped — for symmetric objects (bowls,
    vases) the yaw-stability score is near-flat across orientations, and a diagonal PCA candidate can
    win by a sliver and *tilt* an already-upright object (observed on a vase). Axis-aligned candidates
    avoid that failure while still correcting the genuine flips/tips (to the nearest axis).
    """
    return [np.eye(3)[i] * sg for i in range(3) for sg in (1.0, -1.0)]


def _iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    inter = ((a > thr) & (b > thr)).sum()
    union = ((a > thr) | (b > thr)).sum()
    return float(inter / max(union, 1))


def _stable_up(points: np.ndarray) -> np.ndarray | None:
    """Geometric 'up' from the most probable resting pose of the convex hull (gravity prior).

    For rotationally-symmetric objects (bowls, vases) under orientation-tolerant affordances
    (contain), GEAL's yaw-stability is near-flat, so this physical prior breaks the tie toward the
    upright (base-down) pose. Returns None if the computation fails (then yaw-stability alone is used).
    """
    try:
        import trimesh
        hull = trimesh.Trimesh(vertices=np.asarray(points, np.float64)).convex_hull
        transforms, probs = trimesh.poses.compute_stable_poses(hull, n_samples=1)
        if transforms is None or len(transforms) == 0:
            return None
        R = transforms[int(np.argmax(probs))][:3, :3]   # maps original -> resting (up = +Z)
        up = R.T @ np.array([0.0, 0.0, 1.0])
        return up / (np.linalg.norm(up) + 1e-12)
    except Exception:
        return None


def find_canonical_rotation(
    points: np.ndarray,
    labeler,
    object_class: str,
    affordance: str,
    *,
    yaws: tuple[float, ...] = (0.0, 90.0, 210.0),
    question: str | None = None,
    question_csv=None,
) -> tuple[np.ndarray, dict]:
    """Find ``R`` (3,3 float32) such that ``points @ R.T`` is the upright/canonical orientation.

    For each candidate up-axis, GEAL is run at several yaws and scored by mean pairwise IoU of the
    thresholded fields (yaw-stability). The most yaw-stable candidate is returned. Cost is
    ``len(candidates) * len(yaws)`` GEAL forward passes (cheap on CPU for ~2048 points).
    """
    pts = np.asarray(points, np.float32)
    stable_up = _stable_up(pts)          # geometric upright prior (or None)
    GEOM_W = 0.2                         # weight; lets GEAL dominate when discriminative (e.g. chair)
    best_score, best_R, best_u, best_yaw = -1.0, np.eye(3, dtype=np.float32), np.array([0, 1, 0.0]), 0.0
    for u in _candidate_ups(pts):
        R = _basis_to_up(u)
        preds = [
            labeler.predict(
                (pts @ R.T @ _yaw(d).T).astype(np.float32),
                object_class, affordance, question=question, question_csv=question_csv,
            )
            for d in yaws
        ]
        ious = [_iou(preds[i], preds[j]) for i in range(len(preds)) for j in range(i + 1, len(preds))]
        yaw_iou = float(np.mean(ious)) if ious else 0.0
        geom = abs(float(u @ stable_up)) if stable_up is not None else 0.0  # align to vertical (either sign)
        score = yaw_iou + GEOM_W * geom
        if score > best_score:
            best_score, best_R, best_u, best_yaw = score, R, u, yaw_iou
    return best_R, {
        "yaw_iou": round(best_yaw, 4),
        "up_axis": np.round(best_u, 3).tolist(),
        "stable_up": np.round(stable_up, 3).tolist() if stable_up is not None else None,
    }
