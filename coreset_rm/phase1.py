"""Phase 1 — constructs the intermediate coreset S_1.

Provides two entry points:
  - run_phase1:      real Phase 1, Steps 1–5 as described in the paper.
  - run_phase1_mock: bypass that returns all n points with unit weight
                     and k-means++ centers. Kept for baseline comparisons.
"""
from __future__ import annotations

import numpy as np

from coreset_rm.alg_g import AlgGOutput
from coreset_rm.config import PipelineConfig
from coreset_rm.dmax_step import compute_d_max_per_cluster
from coreset_rm.instrumentation import Metrics, track
from coreset_rm.kmeans import kmeans_plusplus
from coreset_rm.oracles import DistanceOracle, QuadrupletOracle
from coreset_rm.prob_sort import prob_sort
from coreset_rm.ring_sampling import run_step5
from coreset_rm.rings import classify_rings
from coreset_rm.rng import RNG
from coreset_rm.step1 import run_step1
from coreset_rm.types import Phase1Output, WeightedPointSet
from coreset_rm.config import D_DISLOCATION, L_RINGS_PER_CLUSTER, S_SAMPLES_PER_RING


def run_phase1(
    n: int,
    k: int,
    dist_oracle: DistanceOracle,
    quad_oracle: QuadrupletOracle,
    root_rng: RNG,
    metrics: Metrics | None = None,
    config: PipelineConfig | None = None,
    algoutput: AlgGOutput | None = None,
) -> Phase1Output:
    if config is None:
        dislocation = D_DISLOCATION
        L_rings = L_RINGS_PER_CLUSTER
        s_samples = S_SAMPLES_PER_RING
    else:
        dislocation = config.D
        L_rings = config.L
        s_samples = config.s_per_ring

    """Run Phase 1, Steps 1–5. Records per-step metrics if `metrics` is given."""
    _noop = Metrics() if metrics is None else metrics

    with track(_noop, "step1_alg_g_and_extract", dist=dist_oracle, quad=quad_oracle):
        partition = run_step1(
            n, k, dist_oracle, quad_oracle, root_rng.spawn("step1"), config, algoutput
        )

    with track(_noop, "step2_prob_sort", dist=dist_oracle, quad=quad_oracle):
        orderings = prob_sort(
            partition.clusters, partition.centers, quad_oracle,
        )

    with track(_noop, "step3_d_max", dist=dist_oracle, quad=quad_oracle):
        d_max_list = compute_d_max_per_cluster(
            orderings, partition.centers, dist_oracle, dislocation
        )

    with track(_noop, "step4_classify_rings", dist=dist_oracle, quad=quad_oracle):
        decompositions = []
        for pi, ci, d_max in zip(orderings, partition.centers, d_max_list):
            decompositions.append(
                classify_rings(
                    ordering=pi, center=ci, d_max=d_max,
                    dist_oracle=dist_oracle, D=dislocation, L=L_rings,
                )
            )

    with track(_noop, "phase2_prep_distances", dist=dist_oracle, quad=quad_oracle):
        return run_step5(
            ring_decompositions=decompositions,
            centers=partition.centers,
            dist_oracle=dist_oracle,
            rng=root_rng.spawn("step5").generator(),
            s_per_ring=s_samples
        )


def run_phase1_mock(
    n: int,
    k: int,
    dist_oracle: DistanceOracle,
    root_rng: RNG,
) -> Phase1Output:
    """Mock Phase 1: k-means++ centers + nearest-center assignment over all n points.

    Used as a baseline comparison against run_phase1. S_1 = P with unit weights.
    """
    centers, _ = kmeans_plusplus(
        n=n, k=k, dist_oracle=dist_oracle,
        rng=root_rng.spawn("phase1_mock_kmeans").generator(),
    )

    all_pts = list(range(n))
    dist_matrix = np.empty((k, n), dtype=np.float64)
    for j, c in enumerate(centers):
        dist_matrix[j] = dist_oracle.distance_one_to_many(c, all_pts)

    assignments = dist_matrix.argmin(axis=0).astype(np.int64)
    center_dists = dist_matrix[assignments, np.arange(n)]

    return Phase1Output(
        S1=WeightedPointSet(indices=all_pts, weights=[1.0] * n),
        centers=centers,
        assignments=assignments.tolist(),
        center_dists=center_dists,
    )