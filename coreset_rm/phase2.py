"""Phase 2 — sensitivity sampling to compress S_1 down to the final coreset.

Pure computation. No oracle queries. Takes Phase1Output and returns a
smaller WeightedPointSet.
"""
from __future__ import annotations
from typing import List

import numpy as np

from coreset_rm.config import Z, M_MULT_PHASE2
from coreset_rm.types import Phase1Output
from coreset_rm.types import WeightedPointSet


def run_phase2(
    phase1_out: Phase1Output,
    m: int,
    rng: np.random.Generator,
) -> WeightedPointSet:
    """Compress S_1 to m weighted points via sensitivity sampling.

    For each point i in cluster c, the sensitivity is:
        sensitivity_i = cost_i / total_cost_c  +  1 / |cluster c|
    where cost_i = w_i * d(i, assigned_center)^z.

    Points are sampled with replacement proportional to sensitivity, and
    reweighted by w_i / (m * p_i) so the coreset preserves total mass.
    """
    indices = np.asarray(phase1_out.S1.indices, dtype=np.int64)
    weights = np.asarray(phase1_out.S1.weights, dtype=np.float64)
    assignments = np.asarray(phase1_out.assignments, dtype=np.int64)
    center_dists = np.asarray(phase1_out.center_dists, dtype=np.float64)

    n_s1 = len(indices)
    if n_s1 == 0:
        raise ValueError("S_1 is empty")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    k = int(assignments.max()) + 1

    # Per-point cost: w_i * d(i, c_i)^z.
    if Z == 2:
        per_point_cost = weights * center_dists ** 2
    else:
        per_point_cost = weights * center_dists ** Z

    # Per-cluster totals (cost and size).
    total_cost_per_cluster = np.zeros(k, dtype=np.float64)
    size_per_cluster = np.zeros(k, dtype=np.int64)
    np.add.at(total_cost_per_cluster, assignments, per_point_cost)
    np.add.at(size_per_cluster, assignments, 1)

    # Sensitivity components.
    # cost_term: cost_i / total_cost_c. Zero if the cluster's total cost is 0.
    cluster_totals = total_cost_per_cluster[assignments]
    cost_term = np.where(
        cluster_totals > 0.0,
        per_point_cost / np.where(cluster_totals > 0.0, cluster_totals, 1.0),
        0.0,
    )
    # uniform_term: 1 / |cluster c|.
    uniform_term = 1.0 / size_per_cluster[assignments]

    sensitivity = cost_term + uniform_term

    # Normalize to a probability distribution.
    total_sensitivity = sensitivity.sum()
    if total_sensitivity <= 0.0:
        # Shouldn't happen: uniform_term is always > 0 for any non-empty cluster.
        raise RuntimeError("all sensitivities are zero — cannot sample")
    probs = sensitivity / total_sensitivity

    # Sample m points with replacement.
    sampled_positions = rng.choice(n_s1, size=m, replace=True, p=probs)

    # Importance-sampling reweight: w_i / (m * p_i).
    sampled_weights = weights[sampled_positions] / (m * probs[sampled_positions])
    sampled_indices = indices[sampled_positions]

    return WeightedPointSet(
        indices=sampled_indices.tolist(),
        weights=sampled_weights.tolist(),
    )


def default_m(k: int) -> int:
    """Default Phase 2 target size from config."""
    return M_MULT_PHASE2 * k