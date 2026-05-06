"""Step 3: d_max^(i) := max_{p in P_i} d(p, c_i) per cluster."""
from __future__ import annotations
from typing import List

from coreset_rm.config import D_DISLOCATION
from coreset_rm.oracles import DistanceOracle


def compute_d_max_per_cluster(
    orderings: List[List[int]],
    centers: List[int],
    dist_oracle: DistanceOracle,
    D: int = D_DISLOCATION,
) -> List[float]:
    """For each cluster, query the last D+1 positions of π_i and take max.

    Returns the per-cluster d_max values.
    """
    k = len(centers)
    if len(orderings) != k:
        raise ValueError(f"got {len(orderings)} orderings but {k} centers")

    d_max_per_cluster: List[float] = []

    for pi, c in zip(orderings, centers):
        n_i = len(pi)
        if n_i == 0:
            d_max_per_cluster.append(0.0)
            continue

        start = max(0, n_i - (D + 1))
        pts = [pi[p] for p in range(start, n_i)]
        dists = dist_oracle.distance_one_to_many(c, pts)
        d_max_per_cluster.append(float(dists.max()))

    return d_max_per_cluster