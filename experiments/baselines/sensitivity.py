"""Sensitivity sampling baseline (Feldman-Langberg 2011).

Runs a k-means++ O(1)-approximate clustering on full P, then applies
the standard FL11 sensitivity sampling to compress to m points.

Mathematically equivalent to ``run_phase1_mock`` followed by
``run_phase2`` (Phase 2 already computes the FL11 sensitivity
``cost_i/total_cluster_cost + 1/cluster_size`` and samples with
replacement), so we compose those two rather than duplicate the
formula.

Caching model matches ``ours``: the kmeans++ clustering of full P
(plus the follow-up assignment pass inside ``run_phase1_mock``) runs
once per seed via ``prepare``; ``run_phase2`` runs per sweep point
via ``build``.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass

import numpy as np

from coreset_rm.config import PipelineConfig
from experiments.baselines.base import BaselineResult
from coreset_rm.oracles import DistanceOracle
from coreset_rm.phase1 import run_phase1_mock
from coreset_rm.phase2 import run_phase2
from coreset_rm.rng import RNG
from coreset_rm.types import Phase1Output

logger = logging.getLogger(__name__)


@dataclass
class CachedSensitivity:
    """Cached kmeans++ clustering of full P, ready for repeated Phase 2
    compressions across a sweep of target sizes.
    """

    phase1_out: Phase1Output
    construction_time_s: float
    distance_queries: int
    root_rng: RNG


def prepare(
    n: int,
    k: int,
    dist_oracle: DistanceOracle,
    root_rng: RNG,
    config: PipelineConfig | None = None,
) -> CachedSensitivity:
    """Run weighted k-means++ on full P and cache the resulting
    Phase 1 output (centers, assignments, per-point center distances).

    Parameters
    ----------
    n : int
        Size of the full dataset P.
    k : int
        Number of clusters for the k-means++ step.
    dist_oracle : DistanceOracle
        Euclidean distance oracle over P. Its ``query_count`` is read
        before and after the kmeans++ + assignment pass to attribute
        usage to this call.
    root_rng : RNG
        Root RNG. A child is spawned inside ``run_phase1_mock`` for
        the kmeans++ draw; ``build`` spawns from this root for each m.
    config : PipelineConfig, optional
        Accepted for API symmetry with the other baselines. Unused;
        ``run_phase1_mock`` takes no config.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    d_before = dist_oracle.query_count
    t0 = time.perf_counter()

    phase1_out = run_phase1_mock(
        n=n, k=k, dist_oracle=dist_oracle, root_rng=root_rng,
    )

    elapsed = time.perf_counter() - t0
    d_used = dist_oracle.query_count - d_before

    logger.info(
        "sensitivity.prepare n=%d k=%d dq=%d time=%.3fs",
        n, k, d_used, elapsed,
    )

    return CachedSensitivity(
        phase1_out=phase1_out,
        construction_time_s=elapsed,
        distance_queries=d_used,
        root_rng=root_rng,
    )


def build(
    coords: np.ndarray,
    cached: CachedSensitivity,
    m: int,
) -> BaselineResult:
    """Run Phase 2 (FL11 sensitivity sampling) with target ``m`` on
    the cached full-data clustering.

    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        Input points. Used only to look up coordinates of sampled
        indices; Phase 2 itself does no oracle queries.
    cached : CachedSensitivity
        Output of ``prepare``. Its time and distance-query counts are
        attributed to the returned ``BaselineResult``.
    m : int
        Target coreset size. If ``m > |S1|`` (i.e. m > n), Phase 2
        samples with replacement and the result contains duplicates.
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    n_s1 = len(cached.phase1_out.S1)
    if m > n_s1:
        logger.warning(
            "sensitivity.build m=%d exceeds |S1|=%d; coreset will "
            "contain duplicates", m, n_s1,
        )

    # Spawn a per-m child RNG so different sweep points at the same
    # seed draw independent samples (same-(seed, m) is reproducible).
    phase2_rng = cached.root_rng.spawn(f"sensitivity_sampling_m{m}").generator()

    t0 = time.perf_counter()
    s2 = run_phase2(
        phase1_out=cached.phase1_out,
        m=m,
        rng=phase2_rng,
    )
    phase2_time = time.perf_counter() - t0

    s2_indices = np.asarray(s2.indices, dtype=np.int64)
    s2_weights = np.asarray(s2.weights, dtype=np.float64)
    sampled_coords = coords[s2_indices]

    total_time = cached.construction_time_s + phase2_time

    logger.info(
        "sensitivity.build m=%d |S1|=%d time=%.3fs (prepare=%.3fs phase2=%.3fs)",
        m, n_s1, total_time, cached.construction_time_s, phase2_time,
    )

    return BaselineResult(
        coreset_coords=sampled_coords,
        coreset_weights=s2_weights,
        construction_time_s=total_time,
        distance_queries=cached.distance_queries,
        quadruplet_queries=0,
    )
