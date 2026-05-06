"""Lightweight coresets (Bachem, Lucic, Krause, NeurIPS 2018).

Samples each point with probability
    p_i = (1 / 2n) + (1/2) * d(x_i, mu)^2 / sum_j d(x_j, mu)^2
where mu is the mean of all data points. Sampled points are reweighted
by w_i = 1 / (m * p_i) for an unbiased cost estimator.

Think of it as "sensitivity sampling against a single center (the
mean)". Cheaper than full sensitivity (O(n) distance computations
vs O(kn)) and better than uniform because it accounts for global
geometry — outliers get higher sampling probability.

Distance-query accounting
-------------------------
One pass over P to compute the mean, then one pass to compute squared
distances to the mean. Strictly, these are "distances from an indexed
point to a non-indexed point (mu)", which don't map to our
DistanceOracle abstraction (which queries between two indexed points
in P). We compute directly from ``coords`` and report
``distance_queries = n`` to reflect the algorithm's distance-computation
budget for Exp 2's companion table. For a real distance-query plot
(Exp 1), the same number is the natural entry.

No prepare/build split
----------------------
Like uniform, lightweight has no expensive precomputation worth caching
across a sweep — the mean + distance pass is O(n) work per call, which
is cheap enough to redo per m.
"""
from __future__ import annotations
import logging
import time

import numpy as np

from experiments.baselines.base import BaselineResult
from coreset_rm.rng import RNG

logger = logging.getLogger(__name__)


def build(
    coords: np.ndarray,
    m: int,
    root_rng: RNG,
) -> BaselineResult:
    """Build a lightweight coreset on ``coords``.

    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        Input points.
    m : int
        Target coreset size. Sampling is with replacement, so the
        returned coreset has exactly ``m`` rows (possibly with
        duplicates).
    root_rng : RNG
        Root RNG. A named child is spawned per m so different sweep
        points draw independently and the same (root seed, m) pair
        is reproducible.

    Returns
    -------
    BaselineResult
        ``coreset_coords`` of shape (m, d) — rows of ``coords`` sampled
        with replacement proportional to the lightweight distribution.
        ``coreset_weights`` are 1 / (m * p_i) (Horvitz-Thompson style).
        ``distance_queries = n`` (see module docstring).
        ``quadruplet_queries = 0``.
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    n = coords.shape[0]

    t0 = time.perf_counter()

    # Mean over all points (not a distance query, just an averaging pass).
    mu = coords.mean(axis=0)

    # Squared distances to mean. This is the O(n) distance-computation
    # pass we report on the x-axis.
    diffs = coords - mu
    sq_dists = np.einsum("ij,ij->i", diffs, diffs)
    total = float(sq_dists.sum())

    if total <= 0.0:
        # Degenerate: every point equals the mean. Fall back to uniform.
        # (No sane dataset hits this, but we handle it for robustness.)
        probs = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        probs = 0.5 / n + 0.5 * sq_dists / total

    rng = root_rng.spawn(f"lightweight_m{m}").generator()
    sampled_positions = rng.choice(n, size=m, replace=True, p=probs)
    sampled_coords = coords[sampled_positions]
    sampled_weights = 1.0 / (m * probs[sampled_positions])

    elapsed = time.perf_counter() - t0

    logger.info(
        "lightweight.build n=%d m=%d dq=%d time=%.3fs weights_sum=%.1f",
        n, m, n, elapsed, sampled_weights.sum(),
    )

    return BaselineResult(
        coreset_coords=sampled_coords,
        coreset_weights=sampled_weights,
        construction_time_s=elapsed,
        distance_queries=n,
        quadruplet_queries=0,
    )
