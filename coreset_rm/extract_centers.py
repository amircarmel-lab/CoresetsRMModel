"""Extract exactly k centers from the Coreset+ (paper's Mettu–Plaxton step).

The paper runs Mettu–Plaxton on the weighted set (C, w) with w(c) = |M^{-1}(c)|.
We substitute k-means++ on C, which gives an O(log k)-approximation instead
of MP's O(1) — good enough to unblock downstream steps.

After picking k centers from C, each point p in P is assigned to the center
nearest to M(p) (from Alg-G), not to p itself. Triangle inequality absorbs
the mapping cost. This keeps the distance-query cost at O(k · |C|) —
independent of n — which is the paper's contribution.
"""
from __future__ import annotations
from typing import List

from coreset_rm.alg_g import AlgGOutput
from coreset_rm.kmeans import kmeans_plusplus
from coreset_rm.oracles import DistanceOracle
from coreset_rm.rng import RNG
from coreset_rm.types import ClusterPartition


def extract_centers(
    alg_g_out: AlgGOutput,
    n: int,
    k: int,
    dist_oracle: DistanceOracle,
    root_rng: RNG,
) -> ClusterPartition:
    """Pick k centers from C via k-means++ over C, then assign all of P."""
    C = alg_g_out.C
    if len(C) < k:
        raise ValueError(f"|C| = {len(C)} < k = {k}; cannot extract k centers")

    # k-means++ over C. Returns:
    #   local_center_indices[0..k-1]: indices into C of the chosen centers.
    #   C_to_center_local[0..|C|-1]: for each c in C, which of the k chosen
    #                                centers it's nearest to.
    C_oracle = _CIndexedDistanceOracle(C, dist_oracle)
    local_center_indices, C_to_center_local = kmeans_plusplus(
        n=len(C), k=k, dist_oracle=C_oracle,
        rng=root_rng.spawn("extract_centers_kmpp").generator(),
    )
    centers = [C[i] for i in local_center_indices]

    # Compose through Alg-G's M: each p in P maps to centers[ C_to_center_local[M[p]] ].
    # No distance queries.
    M = alg_g_out.M
    assignment = [int(C_to_center_local[M[i]]) for i in range(n)]

    clusters: List[List[int]] = [[] for _ in range(k)]
    for i, c_idx in enumerate(assignment):
        clusters[c_idx].append(i)

    return ClusterPartition(
        centers=centers,
        assignment=assignment,
        clusters=clusters,
    )


class _CIndexedDistanceOracle:
    """Adapter: exposes C as points 0..|C|-1 backed by the real distance oracle."""

    __slots__ = ("_C", "_inner", "query_count")

    def __init__(self, C: List[int], inner: DistanceOracle):
        self._C = C
        self._inner = inner
        self.query_count = 0

    def distance(self, u: int, v: int) -> float:
        self.query_count += 1
        return self._inner.distance(self._C[u], self._C[v])

    def distance_one_to_many(self, u: int, vs):
        self.query_count += len(vs)
        real_u = self._C[u]
        real_vs = [self._C[v] for v in vs]
        return self._inner.distance_one_to_many(real_u, real_vs)