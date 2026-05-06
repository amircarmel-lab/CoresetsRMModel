"""Comparator-driven stand-ins for ProbSort (Lemma 3).

Two variants used by the pipeline:
  - prob_sort:       sorts each cluster by distance-to-center (phase1).
  - prob_sort_edges: sorts a flat edge list globally (alg_g).
"""
from __future__ import annotations
from functools import cmp_to_key
from typing import List, Tuple

from coreset_rm.oracles import QuadrupletOracle


def prob_sort(
    clusters: List[List[int]],
    centers: List[int],
    quad_oracle: QuadrupletOracle,
) -> List[List[int]]:
    k = len(centers)
    if len(clusters) != k:
        raise ValueError(f"got {len(clusters)} clusters but {k} centers")

    orderings: List[List[int]] = []
    for cluster_pts, c in zip(clusters, centers):
        if len(cluster_pts) <= 1:
            orderings.append(list(cluster_pts))
            continue

        # c=c captures the loop variable by value for the closure.
        def compare(p: int, q: int, c=c) -> int:
            return -1 if quad_oracle.compare(c, p, c, q) else 1

        orderings.append(sorted(cluster_pts, key=cmp_to_key(compare)))

    return orderings


def prob_sort_edges(
    edges: List[Tuple[int, int]],
    quad_oracle: QuadrupletOracle,
) -> List[Tuple[int, int]]:
    """Sort a flat list of edges in ascending distance order via merge sort."""
    if len(edges) <= 1:
        return list(edges)

    mid = len(edges) // 2
    left = prob_sort_edges(edges[:mid], quad_oracle)
    right = prob_sort_edges(edges[mid:], quad_oracle)
    return _merge_edges(left, right, quad_oracle)


def _merge_edges(
    left: List[Tuple[int, int]],
    right: List[Tuple[int, int]],
    quad_oracle: QuadrupletOracle,
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        u1, v1 = left[i]
        u2, v2 = right[j]
        if quad_oracle.compare(u1, v1, u2, v2):
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out