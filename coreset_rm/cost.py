"""Ground-truth weighted k-means cost evaluation.

NOT PART OF THE PIPELINE. Reads coordinates directly, bypassing oracles.
Used by tests and experiments as the judge — never by algorithm code.
Algorithm code must only see point indices through DistanceOracle and
QuadrupletOracle.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

from coreset_rm.config import Z


def weighted_kmeans_cost(
    points: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """sum_i w_i * min_c ||p_i - c||^z.

    points:  (n, d) coordinates of points being evaluated.
    centers: (k, d) coordinates of centers. These may or may not be rows
             of `points` — Lloyd iterations produce means, not input points.
    weights: length-n non-negative weights; None means uniform weight 1.
    """
    if points.ndim != 2 or centers.ndim != 2:
        raise ValueError("points and centers must be 2D arrays")
    if points.shape[1] != centers.shape[1]:
        raise ValueError("points and centers must have matching dimension")

    sq = np.sum(points**2, axis=1, keepdims=True) \
       + np.sum(centers**2, axis=1)[None, :] \
       - 2.0 * (points @ centers.T)
    np.maximum(sq, 0.0, out=sq)

    min_sq = np.min(sq, axis=1)
    per_point = min_sq if Z == 2 else min_sq ** (Z / 2.0)

    if weights is None:
        return float(per_point.sum())
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (points.shape[0],):
        raise ValueError(
            f"weights shape {weights.shape} does not match n={points.shape[0]}"
        )
    return float((per_point * weights).sum())