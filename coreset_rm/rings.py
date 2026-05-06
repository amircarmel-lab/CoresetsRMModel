"""Step 4: ring decomposition (Algorithm 1 from the paper).

Given an approximately-sorted ordering π_i with max dislocation D,
classify every point in the cluster into one of L rings or the close
set, using O(L · (log n + D)) exact distance queries.

Rings are geometric (ratio-2):
    thresholds[ℓ] = d_max^z / 2^(L-ℓ),   ℓ = 0..L
    close set:  d^z ≤ thresholds[0]
    ring ℓ (1-indexed):  thresholds[ℓ-1] < d^z ≤ thresholds[ℓ]

A local distance cache is maintained inside this function to avoid
re-querying the same position across overlapping ambiguity zones and
repeated binary-search probes. The cache is internal — it does not
leak out of classify_rings.
"""
from __future__ import annotations
from typing import Dict, List

from coreset_rm.config import D_DISLOCATION, L_RINGS_PER_CLUSTER, Z
from coreset_rm.oracles import DistanceOracle
from coreset_rm.types import RingDecomposition


def classify_rings(
    ordering: List[int],
    center: int,
    d_max: float,
    dist_oracle: DistanceOracle,
    D: int = D_DISLOCATION,
    L: int = L_RINGS_PER_CLUSTER,
) -> RingDecomposition:
    """Classify every point in `ordering` into close or one of L rings."""
    n_i = len(ordering)

    # Trivial cases.
    if n_i == 0:
        return RingDecomposition(
            close=[], rings=[[] for _ in range(L)],
            thresholds=[0.0] * (L + 1), d_max=0.0,
        )
    if d_max == 0.0:
        return RingDecomposition(
            close=list(ordering), rings=[[] for _ in range(L)],
            thresholds=[0.0] * (L + 1), d_max=0.0,
        )

    # Build the L+1 thresholds: θ[ℓ] = d_max^z / 2^(L-ℓ).
    d_max_z = d_max ** Z
    thresholds = [d_max_z / (2 ** (L - ell)) for ell in range(L + 1)]

    # Local cache: position-in-ordering → distance. Scoped to this call.
    cache: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Step 4.1: binary-search for each threshold's boundary position.
    # ------------------------------------------------------------------
    boundary_positions: List[int] = []
    for theta in thresholds:
        j = _binary_search_threshold(
            ordering, center, theta, dist_oracle, cache,
        )
        boundary_positions.append(j)

    # ------------------------------------------------------------------
    # Step 4.2: resolve ambiguity zones around each boundary.
    # Cache prevents double-querying when zones overlap.
    # ------------------------------------------------------------------
    for j_l in boundary_positions:
        lo = max(0, j_l - D)
        hi = min(n_i, j_l + D + 1)
        unresolved_positions = [p for p in range(lo, hi) if p not in cache]
        if unresolved_positions:
            pts = [ordering[p] for p in unresolved_positions]
            dists = dist_oracle.distance_one_to_many(center, pts)
            for p, d in zip(unresolved_positions, dists):
                cache[p] = float(d)

    # ------------------------------------------------------------------
    # Step 4.3: assign each point to close or a ring.
    # ------------------------------------------------------------------
    close: List[int] = []
    rings: List[List[int]] = [[] for _ in range(L)]

    for pos in range(n_i):
        pt = ordering[pos]
        if pos in cache:
            d_z = cache[pos] ** Z
            bucket = _bucket_by_distance(d_z, thresholds)
        else:
            bucket = _bucket_by_position(pos, boundary_positions, D)

        if bucket == 0:
            close.append(pt)
        else:
            rings[bucket - 1].append(pt)

    return RingDecomposition(
        close=close, rings=rings,
        thresholds=thresholds, d_max=d_max,
    )


def _binary_search_threshold(
    ordering: List[int],
    center: int,
    threshold_z: float,
    dist_oracle: DistanceOracle,
    cache: Dict[int, float],
) -> int:
    """Return the first position p such that d(ordering[p], center)^z > threshold.

    Equivalently: the count of positions at or below the threshold.
    Because of dislocation, the returned position is within D of the true
    boundary — that's resolved by Step 4.2.
    """
    lo, hi = 0, len(ordering)
    while lo < hi:
        mid = (lo + hi) // 2
        if mid in cache:
            d = cache[mid]
        else:
            d = float(dist_oracle.distance(center, ordering[mid]))
            cache[mid] = d
        if d ** Z <= threshold_z:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bucket_by_distance(d_z: float, thresholds: List[float]) -> int:
    """Return the bucket index: 0 = close, ℓ = ring ℓ (1-indexed).

    bucket 0: d_z ≤ thresholds[0]
    bucket ℓ: thresholds[ℓ-1] < d_z ≤ thresholds[ℓ]
    """
    if d_z <= thresholds[0]:
        return 0
    for ell in range(1, len(thresholds)):
        if d_z <= thresholds[ell]:
            return ell
    # d_z exceeds d_max^z (shouldn't happen but be defensive).
    return len(thresholds) - 1


def _bucket_by_position(pos: int, boundary_positions: List[int], D: int) -> int:
    """Classify by position when distance is unknown.

    Returns the unique bucket ℓ such that boundary_positions[ℓ] > pos + D.
    This is valid only when pos is outside every ambiguity zone, which is
    exactly when the distance wasn't queried.
    """
    for ell, j_l in enumerate(boundary_positions):
        if pos < j_l - D:
            return ell
    # Past the last boundary — outermost ring.
    return len(boundary_positions) - 1