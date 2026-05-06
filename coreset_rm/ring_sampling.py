"""Step 5: per-ring uniform sampling, assemble S_1.

For each cluster i:
  - Center c_i gets weight |close_i| (mass of all points snapped to it).
  - Each ring ℓ contributes min(|ring|, s) uniformly-sampled points,
    each weighted |ring| / samples_taken. If |ring| ≤ s, the ring is
    taken whole with weight 1 per point.

Distances from each sampled S_1 point to its assigned center are queried
fresh (batched per cluster) and returned alongside.
"""
from __future__ import annotations
from typing import List

import numpy as np

from coreset_rm.config import S_SAMPLES_PER_RING
from coreset_rm.oracles import DistanceOracle
from coreset_rm.types import Phase1Output
from coreset_rm.types import RingDecomposition, WeightedPointSet


def run_step5(
    ring_decompositions: List[RingDecomposition],
    centers: List[int],
    dist_oracle: DistanceOracle,
    rng: np.random.Generator,
    s_per_ring: int = S_SAMPLES_PER_RING,
) -> Phase1Output:
    k = len(centers)
    if len(ring_decompositions) != k:
        raise ValueError(
            f"got {len(ring_decompositions)} decompositions but {k} centers"
        )

    indices: List[int] = []
    weights: List[float] = []
    assignments: List[int] = []

    for i, (ci, rd) in enumerate(zip(centers, ring_decompositions)):
        indices.append(ci)
        weights.append(float(len(rd.close)))
        assignments.append(i)

        # Ring entries: uniform sample per ring.
        for ring in rd.rings:
            if not ring:
                continue
            if len(ring) <= s_per_ring:
                for pt in ring:
                    indices.append(int(pt))
                    weights.append(1.0)
                    assignments.append(i)
            else:
                sampled = rng.choice(ring, size=s_per_ring, replace=False)
                weight_each = len(ring) / s_per_ring
                for pt in sampled:
                    indices.append(int(pt))
                    weights.append(weight_each)
                    assignments.append(i)

    # Batched distance queries per cluster.
    center_dists = np.zeros(len(indices), dtype=np.float64)
    for i, ci in enumerate(centers):
        rows = [row for row, a in enumerate(assignments) if a == i]
        if not rows:
            continue
        pts = [indices[row] for row in rows]
        ds = dist_oracle.distance_one_to_many(ci, pts)
        for row, d in zip(rows, ds):
            center_dists[row] = d

    return Phase1Output(
        S1=WeightedPointSet(indices=indices, weights=weights),
        centers=list(centers),
        assignments=assignments,
        center_dists=center_dists,
    )