"""Alg-G-as-coreset baseline.

Runs Alg-G (quadruplet-only) on P to obtain the natural Coreset+
(C, w_M) of size O(k polylog n), then importance-samples from it
proportional to w_M to produce a coreset of target size m.

Caching model
-------------
Alg-G is expensive (O(kn polylog n) quadruplet queries), and its
randomness contributes negligibly compared to the variance of the
downstream subsample. For a coreset-size sweep where many target
sizes m are evaluated against the same (dataset, k, seed), we run
Alg-G ONCE via ``prepare()`` and then call ``build()`` repeatedly
with different m values sharing the same cached Coreset+.

This is a deliberate choice:

- It keeps the baseline fair across sweep points (same underlying
  Coreset+ for every m), so any cost differences reflect the
  subsample choice rather than Alg-G variance.
- It saves experiment time roughly linearly in the number of
  sweep points, with no scientific loss.

If the caller wants a fresh Alg-G run per build, they can simply
call ``prepare`` before each ``build``.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass

import numpy as np

from coreset_rm.alg_g import AlgGOutput, run_alg_g
from coreset_rm.config import PipelineConfig
from base import BaselineResult
from coreset_rm.oracles import QuadrupletOracle
from coreset_rm.rng import RNG

logger = logging.getLogger(__name__)


@dataclass
class CachedAlgG:
    """Output of a single Alg-G run, plus its attribution metrics.

    Passed from ``prepare()`` to one or more ``build()`` calls so that
    a coreset-size sweep can share one Alg-G run across target sizes.
    """

    output: AlgGOutput
    construction_time_s: float
    quadruplet_queries: int


def prepare(
    n: int,
    k: int,
    quad_oracle: QuadrupletOracle,
    root_rng: RNG,
    config: PipelineConfig | None = None,
    algoutput: AlgGOutput | None = None,
) -> CachedAlgG:
    """Run Alg-G and cache its output for reuse across sweep points.

    Intended to be called once per (dataset, k, seed). The returned
    ``CachedAlgG`` can be passed to ``build`` many times.
    """
    q_before = quad_oracle.query_count
    t0 = time.perf_counter()
    if algoutput is None:
        alg_g_output = run_alg_g(n, k, quad_oracle, root_rng, config=config)
        elapsed = time.perf_counter() - t0
        q_used = quad_oracle.query_count - q_before
        c_size = len(alg_g_output.C)
        logger.info(
            "alg_g_as_coreset.prepare n=%d k=%d |C|=%d qq=%d time=%.3fs",
            n, k, c_size, q_used, elapsed,
        )
    else:
        alg_g_output = algoutput
        elapsed = -9999
        q_used = -9999
        logger.info("used ")
        c_size = len(alg_g_output.C)
        logger.info(
            "alg_g_as_coreset.prepare n=%d k=%d |C|=%d, using prepared coreset",
            n, k, c_size,
        )

    return CachedAlgG(
        output=alg_g_output,
        construction_time_s=elapsed,
        quadruplet_queries=q_used,
    )


def build(
    coords: np.ndarray,
    cached: CachedAlgG,
    m: int,
    root_rng: RNG,
) -> BaselineResult:
    """Subsample ``m`` weighted points from a cached Alg-G Coreset+.

    Points are sampled with replacement from C with probability
    proportional to w_M. Since the sampling distribution is exactly
    w_M / sum(w_M), the importance-sampling reweight
    w_M[i] / (m * p_i) collapses to sum(w_M) / m = n / m for every
    sampled point, so all output weights are n / m.

    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        Input points. Used only to look up coordinates of sampled
        representatives; never read for distance queries.
    cached : CachedAlgG
        Cached Alg-G output from ``prepare``. Its time and quadruplet
        query counts are attributed to the returned ``BaselineResult``.
    m : int
        Target coreset size. If ``m > |C|``, result contains
        duplicates (sampling with replacement).
    root_rng : RNG
        Used to spawn a named child RNG for the subsample draw.

    Returns
    -------
    BaselineResult
        ``coreset_coords`` of shape (m, d); uniform weights n / m;
        ``distance_queries = 0``; time and quadruplet counts include
        the cached Alg-G run plus the subsample cost.
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    n = coords.shape[0]
    if n == 0:
        raise ValueError("coords is empty")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    C = np.asarray(cached.output.C, dtype=np.int64)
    w_M = np.asarray(cached.output.weights, dtype=np.float64)
    c_size = len(C)

    total_weight = float(w_M.sum())
    if total_weight <= 0.0:
        raise RuntimeError("cached Alg-G output has zero total weight")

    if m > c_size:
        logger.warning(
            "alg_g_as_coreset.build m=%d exceeds |C|=%d; coreset will "
            "contain duplicates", m, c_size,
        )

    probs = w_M / total_weight

    rng = root_rng.spawn("alg_g_resample").generator()
    t0 = time.perf_counter()

    sampled_positions = rng.choice(c_size, size=m, replace=True, p=probs)
    sampled_point_ids = C[sampled_positions]
    sampled_coords = coords[sampled_point_ids]
    sampled_weights = w_M[sampled_positions] / (m * probs[sampled_positions])

    subsample_time = time.perf_counter() - t0

    total_time = cached.construction_time_s + subsample_time

    logger.info(
        "alg_g_as_coreset.build m=%d |C|=%d time=%.3fs (prepare=%.3fs subsample=%.3fs)",
        m, c_size, total_time, cached.construction_time_s, subsample_time,
    )

    return BaselineResult(
        coreset_coords=sampled_coords,
        coreset_weights=sampled_weights,
        construction_time_s=total_time,
        distance_queries=0,
        quadruplet_queries=cached.quadruplet_queries,
    )