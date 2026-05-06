"""k-means++ seeding via a distance oracle.

Standalone algorithm. Returns k distinct point indices using the
D²-weighted sampling scheme of Arthur & Vassilvitskii (2007).
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from coreset_rm.oracles import DistanceOracle


def kmeans_plusplus(
    n: int,
    k: int,
    dist_oracle: DistanceOracle,
    rng: np.random.Generator,
) -> Tuple[List[int], np.ndarray]:
    """k-means++ seeding over n points via a distance oracle.

    Returns a tuple:
      centers:          k distinct point indices.
      closest:   length-n array; closest[i] ∈ {0..k-1} is
                        the index into `centers` of the chosen center
                        nearest to point i. Built for free during D² updates.

    Uses k * n distance queries for k >= 2 (zero for k = 1).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"cannot pick k={k} centers from n={n} points")

    all_indices = np.arange(n)

    first = int(rng.integers(0, n))
    centers: List[int] = [first]

    if k == 1:
        return centers, np.zeros(n, dtype=np.int64)

    # D[i] = squared distance from point i to nearest chosen center;
    # closest[i] = index (into `centers`) of that nearest center.
    distances = dist_oracle.distance_one_to_many(first, all_indices.tolist())
    D = distances ** 2
    closest = np.zeros(n, dtype=np.int64)

    for j in range(1, k):
        total = D.sum()
        if total <= 0.0:
            remaining = np.setdiff1d(all_indices, np.array(centers), assume_unique=False)
            next_center = int(rng.choice(remaining))
        else:
            probs = D / total
            next_center = int(rng.choice(n, p=probs))

        centers.append(next_center)

        new_distances = dist_oracle.distance_one_to_many(next_center, all_indices.tolist())
        new_D = new_distances ** 2
        improved = new_D < D
        D = np.where(improved, new_D, D)
        closest = np.where(improved, j, closest)

    return centers, closest