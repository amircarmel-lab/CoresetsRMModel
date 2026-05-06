"""Ours — the full pipeline (Phase 1 + Phase 2) as a coreset baseline.

Caching model
-------------
Phase 1 is expensive: O(kn polylog n) quadruplet queries plus
O~(k^2 · polylog n) distance queries to produce the intermediate
coreset S_1. Phase 2 is cheap pure computation and takes the target
coreset size m as an argument.

For a coreset-size sweep over (dataset, k, seed) we run Phase 1 once
via ``prepare`` and then call ``build`` repeatedly with different m
values sharing the same cached S_1. See ``alg_g_as_coreset`` for the
same pattern; within a seed, only the Phase 2 subsample varies
across sweep points, while seed-to-seed variance across runs includes
every random step in Phase 1.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass

import numpy as np

from coreset_rm.alg_g import AlgGOutput
from coreset_rm.config import PipelineConfig
from coreset_rm.instrumentation import Metrics
from coreset_rm.oracles import DistanceOracle, QuadrupletOracle
from coreset_rm.phase1 import run_phase1
from coreset_rm.phase2 import run_phase2
from coreset_rm.rng import RNG
from coreset_rm.types import Phase1Output
from experiments.baselines.base import BaselineResult

logger = logging.getLogger(__name__)


@dataclass
class CachedPhase1:
    """Output of a single Phase 1 run, plus its attribution metrics.

    Attributes
    ----------
    output : Phase1Output
        The intermediate coreset S_1 and the Phase-1 centers/assignments.
    metrics : Metrics
        Per-step timing and query breakdown recorded by ``run_phase1``.
        Available for detailed inspection; ``prepare`` only logs a
        one-line summary.
    construction_time_s, distance_queries, quadruplet_queries : float, int, int
        Totals for the Phase 1 call, for quick attribution inside
        ``build`` without re-reading ``metrics``.
    root_rng : RNG
        Retained so that ``build`` can spawn Phase 2's RNG from the
        same seeded root, keeping draws reproducible per (seed, m).
    """

    output: Phase1Output
    metrics: Metrics
    construction_time_s: float
    distance_queries: int
    quadruplet_queries: int
    root_rng: RNG


def prepare(
    coords: np.ndarray,
    k: int,
    dist_oracle: DistanceOracle,
    quad_oracle: QuadrupletOracle,
    root_rng: RNG,
    config: PipelineConfig | None = None,
    algoutput: AlgGOutput | None = None,
) -> CachedPhase1:
    """Run Phase 1 and cache S_1 for reuse across sweep points.

    Intended to be called once per (dataset, k, seed). The returned
    ``CachedPhase1`` can be passed to ``build`` many times.
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    n = coords.shape[0]
    if n == 0:
        raise ValueError("coords is empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    metrics = Metrics()
    d_before = dist_oracle.query_count
    q_before = quad_oracle.query_count
    t0 = time.perf_counter()

    phase1_out = run_phase1(
        n=n,
        k=k,
        dist_oracle=dist_oracle,
        quad_oracle=quad_oracle,
        root_rng=root_rng,
        metrics=metrics,
        config=config,
        algoutput=algoutput
    )

    elapsed = time.perf_counter() - t0
    d_used = dist_oracle.query_count - d_before
    q_used = quad_oracle.query_count - q_before

    s1_size = len(phase1_out.S1)
    metrics.intermediate_S1_size = s1_size

    if algoutput is None:
        logger.info(
            "ours.prepare n=%d k=%d |S1|=%d dq=%d qq=%d time=%.3fs",
            n, k, s1_size, d_used, q_used, elapsed,
        )
    else:
        logger.info("ours.prepare, using prepared alg-g coreset, n=%d k=%d |S1|=%d dq=%d qq=%d time=%.3fs",
                    n, k, s1_size, d_used, q_used, elapsed, )

    return CachedPhase1(
        output=phase1_out,
        metrics=metrics,
        construction_time_s=elapsed,
        distance_queries=d_used,
        quadruplet_queries=q_used,
        root_rng=root_rng,
    )


def build(
    coords: np.ndarray,
    cached: CachedPhase1,
    m: int,
) -> BaselineResult:
    """Run Phase 2 with target ``m`` on a cached Phase 1 output.

    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        Input points. Used only to look up coordinates of sampled
        indices; Phase 2 itself does no oracle queries.
    cached : CachedPhase1
        Cached Phase 1 output from ``prepare``. Its time and query
        counts are attributed to the returned ``BaselineResult``; the
        Phase 2 RNG is spawned from the same root.
    m : int
        Target coreset size. If ``m > |S1|``, result contains
        duplicates (Phase 2 samples with replacement).

    Returns
    -------
    BaselineResult
        ``coreset_coords`` of shape (m, d); weights reflect the
        sensitivity-sampling reweight inside Phase 2; time and query
        counts include the cached Phase 1 run plus the (cheap) Phase 2.
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    s1_size = len(cached.output.S1)
    if s1_size == 0:
        raise RuntimeError("cached Phase 1 output has empty S_1")
    if m > s1_size:
        logger.warning(
            "ours.build m=%d exceeds |S1|=%d; coreset will contain duplicates",
            m, s1_size,
        )

    # Spawn a unique child RNG per target m so different sweep points
    # at the same seed draw from independent streams (same-(seed,m)
    # calls are reproducible; different-m draws aren't nested).
    phase2_rng = cached.root_rng.spawn(f"phase2_sampling_m{m}").generator()

    t0 = time.perf_counter()
    s2 = run_phase2(
        phase1_out=cached.output,
        m=m,
        rng=phase2_rng,
    )
    phase2_time = time.perf_counter() - t0

    s2_indices = np.asarray(s2.indices, dtype=np.int64)
    s2_weights = np.asarray(s2.weights, dtype=np.float64)
    sampled_coords = coords[s2_indices]

    total_time = cached.construction_time_s + phase2_time

    logger.info(
        "ours.build m=%d |S1|=%d time=%.3fs (prepare=%.3fs phase2=%.3fs)",
        m, s1_size, total_time, cached.construction_time_s, phase2_time,
    )

    return BaselineResult(
        coreset_coords=sampled_coords,
        coreset_weights=s2_weights,
        construction_time_s=total_time,
        distance_queries=cached.distance_queries,
        quadruplet_queries=cached.quadruplet_queries,
    )