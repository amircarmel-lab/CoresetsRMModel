from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class CoresetPlus:
    """Output of Step 1's approximate clustering (Alg-G).

    C:  list of point indices — the representatives.
    M:  M[i] = index into C that point i maps to. Length n.
    """
    C: List[int]
    M: List[int]


@dataclass
class WeightedPointSet:
    """A weighted set of point indices. Used for S_1 and S_2."""
    indices: List[int]
    weights: List[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.weights):
            raise ValueError(
                f"indices/weights length mismatch: "
                f"{len(self.indices)} vs {len(self.weights)}"
            )

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class ClusterPartition:
    """Output of Step 1's assignment phase.

    centers:     k point indices (the C* from Mettu-Plaxton / k-means++).
    assignment:  assignment[i] ∈ {0,...,k-1} — which center point i maps to.
    clusters:    clusters[j] = list of point indices assigned to centers[j].
    """
    centers: List[int]
    assignment: List[int]
    clusters: List[List[int]]

    @property
    def k(self) -> int:
        return len(self.centers)


@dataclass
class RingDecomposition:
    """Output of Step 4 for a single cluster.

    close:      indices of points snapped to the center (|P_cl^(i)|).
    rings:      rings[l] = indices of points in ring l, l = 0..L-1.
                ring l contains points with
                    thresholds[l] < d(p, c)^z <= thresholds[l+1].
    thresholds: length L+1. thresholds[0] is the close threshold,
                thresholds[L] is d_max^z. ratio-2: thresholds[l+1] = 2*thresholds[l].
    d_max:      max distance from center to any point in this cluster (Step 3).
    """
    close: List[int]
    rings: List[List[int]]
    thresholds: List[float]
    d_max: float


@dataclass
class Phase1Output:
    """Everything Phase 2 needs from Phase 1."""
    S1: WeightedPointSet
    centers: List[int]
    assignments: List[int]
    center_dists: np.ndarray