"""Implementations for the evaluation functions.

Nothing here uses oracles — evaluation reads coordinates directly. Costs
delegate to `coreset_rm.cost.weighted_kmeans_cost`, which is the
project's single source of truth for the (k, z)-clustering cost.

Weighted k-means++ is implemented locally rather than routing through
`coreset_rm.kmeans.kmeans_plusplus`, because the pipeline's version is
oracle-based (counts distance queries) and these runs are evaluation —
their queries must not be counted toward the x-axis.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from coreset_rm.cost import weighted_kmeans_cost


# ----------------------------------------------------------------------
# Weighted k-means++ helper
# ----------------------------------------------------------------------
def _kmeans_plusplus_weighted(
    coords: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weighted k-means++ seeding; returns k center **coordinates**.

    Parameters
    ----------
    coords : (n, d) float array.
    weights : (n,) non-negative float array.
    k : number of centers. Must satisfy 1 <= k <= n.
    rng : numpy Generator.

    Returns
    -------
    centers : (k, d) float array.

    Notes
    -----
    - First center is drawn proportional to `weights`.
    - Subsequent centers are drawn proportional to `weights * D²` where D is
      the distance to the nearest already-chosen center. This is the
      standard weighted generalization of Arthur & Vassilvitskii (2007).
    - When all D² become zero (rare; happens only if remaining points
      coincide exactly with chosen centers), falls back to uniform sampling
      to make progress.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n = coords.shape[0]
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")

    w_total = float(weights.sum())
    if w_total <= 0.0:
        raise ValueError("total weight is zero")

    centers = np.empty((k, coords.shape[1]), dtype=coords.dtype)

    # First center: weighted uniform.
    p0 = weights / w_total
    first_idx = int(rng.choice(n, p=p0))
    centers[0] = coords[first_idx]

    if k == 1:
        return centers

    # min_sq[i] = squared distance from point i to the nearest chosen center.
    min_sq = np.sum((coords - centers[0]) ** 2, axis=1)

    for j in range(1, k):
        sampling_weights = min_sq * weights
        total = float(sampling_weights.sum())
        if total <= 0.0:
            idx = int(rng.integers(0, n))
        else:
            idx = int(rng.choice(n, p=sampling_weights / total))
        centers[j] = coords[idx]

        new_sq = np.sum((coords - centers[j]) ** 2, axis=1)
        np.minimum(min_sq, new_sq, out=min_sq)

    return centers

# ----------------------------------------------------------------------
# Shared evaluation body
# ----------------------------------------------------------------------
def _evaluate_cost_from_coreset_coords(
    coreset_coords: np.ndarray,
    coreset_weights: np.ndarray,
    full_coords: np.ndarray,
    k: int,
    n_kmpp_seeds: int,
    base_seed: int,
) -> np.ndarray:
    """Run weighted k-means++ ``n_kmpp_seeds`` times on the given
    coreset coordinates and return the full-data cost of each run.

    Private helper used by ``evaluate_coreset`` and
    ``evaluate_coreset_coords``. Inputs are assumed already validated
    by the public entry points.
    """
    costs = np.empty(n_kmpp_seeds, dtype=np.float64)
    for i in range(n_kmpp_seeds):
        rng = np.random.default_rng(base_seed + i)
        centers = _kmeans_plusplus_weighted(
            coreset_coords, coreset_weights, k, rng,
        )
        costs[i] = weighted_kmeans_cost(full_coords, centers)
    return costs


# ----------------------------------------------------------------------
# Public: index-based (coreset consists of points from P)
# ----------------------------------------------------------------------
def evaluate_coreset(
    coreset_indices: Sequence[int],
    coreset_weights: Sequence[float],
    full_coords: np.ndarray,
    k: int,
    n_kmpp_seeds: int = 20,
    base_seed: int = 0,
) -> np.ndarray:
    """Run weighted k-means++ on the coreset ``n_kmpp_seeds`` times; return
    the resulting cost on ``full_coords`` for each run.

    Use when the coreset points are drawn from P and are referenced by
    index into ``full_coords``.

    The returned array has shape ``(n_kmpp_seeds,)``. This is Exp 1's
    y-axis source — the caller typically plots mean with a min/max band.

    Parameters
    ----------
    coreset_indices : length-m list of indices into ``full_coords``.
    coreset_weights : length-m list of non-negative weights.
    full_coords : (n, d) float array. Ground-truth coordinates; cost is
                  evaluated against the whole of P.
    k : number of centers.
    n_kmpp_seeds : number of independent k-means++ runs on the coreset.
    base_seed : seed ``i``-th run uses ``base_seed + i``, so two calls
                with the same ``base_seed`` return identical arrays.
    """
    indices = np.asarray(coreset_indices, dtype=np.int64)
    weights = np.asarray(coreset_weights, dtype=np.float64)
    if indices.shape != weights.shape:
        raise ValueError(
            f"indices ({indices.shape}) and weights ({weights.shape}) "
            f"must have the same shape"
        )
    if n_kmpp_seeds < 1:
        raise ValueError(f"n_kmpp_seeds must be >= 1, got {n_kmpp_seeds}")

    coreset_coords = np.ascontiguousarray(full_coords[indices])
    return _evaluate_cost_from_coreset_coords(
        coreset_coords, weights, full_coords, k, n_kmpp_seeds, base_seed,
    )


# ----------------------------------------------------------------------
# Public: coord-based (coreset consists of points possibly outside P)
# ----------------------------------------------------------------------
def evaluate_coreset_coords(
    coreset_coords: np.ndarray,
    coreset_weights: Sequence[float],
    full_coords: np.ndarray,
    k: int,
    n_kmpp_seeds: int = 20,
    base_seed: int = 0,
) -> np.ndarray:
    """Coordinate-based variant of ``evaluate_coreset``.

    Use when the coreset points are not drawn from P — for example,
    BICO's synthesized summary centroids, or any other construction
    whose output lives in the ambient space but is not a subset of P.
    Semantics are identical to ``evaluate_coreset``; the only
    difference is that coordinates are supplied directly rather than
    looked up via indices.

    The returned array has shape ``(n_kmpp_seeds,)``.

    Parameters
    ----------
    coreset_coords : (m, d) float array. Must share the trailing
                     dimension with ``full_coords``.
    coreset_weights : length-m non-negative weights.
    full_coords : (n, d) float array. Ground-truth coordinates; cost is
                  evaluated against the whole of P.
    k : number of centers.
    n_kmpp_seeds : number of independent k-means++ runs on the coreset.
    base_seed : seed ``i``-th run uses ``base_seed + i``, so two calls
                with the same ``base_seed`` return identical arrays.
    """
    coords = np.ascontiguousarray(coreset_coords, dtype=np.float64)
    weights = np.asarray(coreset_weights, dtype=np.float64)

    if coords.ndim != 2:
        raise ValueError(
            f"coreset_coords must be 2-D, got shape {coords.shape}"
        )
    if weights.ndim != 1:
        raise ValueError(
            f"coreset_weights must be 1-D, got shape {weights.shape}"
        )
    if coords.shape[0] != weights.shape[0]:
        raise ValueError(
            f"coreset_coords has m={coords.shape[0]} but "
            f"coreset_weights has m={weights.shape[0]}"
        )
    if coords.shape[1] != full_coords.shape[1]:
        raise ValueError(
            f"coreset_coords has d={coords.shape[1]} but "
            f"full_coords has d={full_coords.shape[1]}"
        )
    if n_kmpp_seeds < 1:
        raise ValueError(f"n_kmpp_seeds must be >= 1, got {n_kmpp_seeds}")

    return _evaluate_cost_from_coreset_coords(
        coords, weights, full_coords, k, n_kmpp_seeds, base_seed,
    )
# ----------------------------------------------------------------------
# Exp 2 / Exp 3A evaluation: worst-case cost ratio
# ----------------------------------------------------------------------
def _evaluate_strong_from_coreset_coords(
    coreset_coords: np.ndarray,
    coreset_weights: np.ndarray,
    full_coords: np.ndarray,
    k: int,
    n_kmpp_seeds: int,
    base_seed: int,
) -> float:
    """Run weighted k-means++ ``n_kmpp_seeds`` times on the **full**
    dataset P (uniform weights) and return the worst-case ratio
    ``|cost(C, P) − cost(C, Q)| / cost(C, P)`` over the seeds.

    Private helper used by ``evaluate_strong_coreset`` and
    ``evaluate_strong_coreset_coords``. Inputs are assumed already
    validated by the public entry points.
    """
    n = full_coords.shape[0]
    uniform_weights = np.ones(n, dtype=np.float64)

    max_ratio = 0.0
    for i in range(n_kmpp_seeds):
        rng = np.random.default_rng(base_seed + i)
        centers = _kmeans_plusplus_weighted(
            full_coords, uniform_weights, k, rng,
        )
        cost_P = weighted_kmeans_cost(full_coords, centers)
        cost_Q = weighted_kmeans_cost(
            coreset_coords, centers, weights=coreset_weights,
        )
        if cost_P == 0.0:
            # Degenerate: every point is a center. Both sides are zero;
            # ratio is defined as zero rather than 0/0.
            ratio = 0.0
        else:
            ratio = abs(cost_P - cost_Q) / cost_P
        if ratio > max_ratio:
            max_ratio = ratio
    return float(max_ratio)


# ----------------------------------------------------------------------
# Public: index-based (coreset consists of points from P)
# ----------------------------------------------------------------------
def evaluate_strong_coreset(
    coreset_indices: Sequence[int],
    coreset_weights: Sequence[float],
    full_coords: np.ndarray,
    k: int,
    n_kmpp_seeds: int = 20,
    base_seed: int = 0,
) -> float:
    """Worst-case cost ratio over ``n_kmpp_seeds`` k-means++ solutions on P.

    For each of ``n_kmpp_seeds`` k-means++ solutions computed on the full
    dataset P, compute the ratio
        |cost(C, P) − cost(C, Q)| / cost(C, P),
    where Q is the weighted coreset and C is the seed's center set.
    Return the **max** across all seeds — the Braverman-style empirical
    ε reported in Exp 2 and Exp 3 Panel A.

    Use when the coreset points are drawn from P and are referenced by
    index into ``full_coords``.

    Parameters
    ----------
    coreset_indices : length-m list of indices into ``full_coords``.
    coreset_weights : length-m list of non-negative weights.
    full_coords : (n, d) float array.
    k : number of centers.
    n_kmpp_seeds : number of independent k-means++ solutions on P.
    base_seed : seed for reproducibility (see ``evaluate_coreset``).
    """
    indices = np.asarray(coreset_indices, dtype=np.int64)
    weights = np.asarray(coreset_weights, dtype=np.float64)
    if indices.shape != weights.shape:
        raise ValueError(
            f"indices ({indices.shape}) and weights ({weights.shape}) "
            f"must have the same shape"
        )
    if n_kmpp_seeds < 1:
        raise ValueError(f"n_kmpp_seeds must be >= 1, got {n_kmpp_seeds}")

    coreset_coords = np.ascontiguousarray(full_coords[indices])
    return _evaluate_strong_from_coreset_coords(
        coreset_coords, weights, full_coords, k, n_kmpp_seeds, base_seed,
    )


# ----------------------------------------------------------------------
# Public: coord-based (coreset consists of points possibly outside P)
# ----------------------------------------------------------------------
def evaluate_strong_coreset_coords(
    coreset_coords: np.ndarray,
    coreset_weights: Sequence[float],
    full_coords: np.ndarray,
    k: int,
    n_kmpp_seeds: int = 20,
    base_seed: int = 0,
) -> float:
    """Coordinate-based variant of ``evaluate_strong_coreset``.

    Use when the coreset points are not drawn from P — for example,
    BICO's synthesized summary centroids, or any other construction
    whose output lives in the ambient space but is not a subset of P.
    Semantics are identical to ``evaluate_strong_coreset``; the only
    difference is that coordinates are supplied directly rather than
    looked up via indices.

    Parameters
    ----------
    coreset_coords : (m, d) float array. Must share the trailing
                     dimension with ``full_coords``.
    coreset_weights : length-m non-negative weights.
    full_coords : (n, d) float array.
    k : number of centers.
    n_kmpp_seeds : number of independent k-means++ solutions on P.
    base_seed : seed for reproducibility.
    """
    coords = np.ascontiguousarray(coreset_coords, dtype=np.float64)
    weights = np.asarray(coreset_weights, dtype=np.float64)

    if coords.ndim != 2:
        raise ValueError(
            f"coreset_coords must be 2-D, got shape {coords.shape}"
        )
    if weights.ndim != 1:
        raise ValueError(
            f"coreset_weights must be 1-D, got shape {weights.shape}"
        )
    if coords.shape[0] != weights.shape[0]:
        raise ValueError(
            f"coreset_coords has m={coords.shape[0]} but "
            f"coreset_weights has m={weights.shape[0]}"
        )
    if coords.shape[1] != full_coords.shape[1]:
        raise ValueError(
            f"coreset_coords has d={coords.shape[1]} but "
            f"full_coords has d={full_coords.shape[1]}"
        )
    if n_kmpp_seeds < 1:
        raise ValueError(f"n_kmpp_seeds must be >= 1, got {n_kmpp_seeds}")

    return _evaluate_strong_from_coreset_coords(
        coords, weights, full_coords, k, n_kmpp_seeds, base_seed,
    )
# ----------------------------------------------------------------------
# Empirical phi: cheap-vs-expensive metric disagreement rate
# ----------------------------------------------------------------------
def measure_empirical_phi(
    cheap_coords: np.ndarray,
    expensive_coords: np.ndarray,
    n_samples: int = 10_000,
    seed: int = 0,
) -> float:
    """Empirical disagreement rate between two metrics on random quadruplets.

    Samples `n_samples` random quadruplets (a, b, c, d) uniformly with
    replacement. For each, asks whether d_cheap(a, b) <= d_cheap(c, d)
    versus d_expensive(a, b) <= d_expensive(c, d). Returns the fraction
    of quadruplets on which the two metrics disagree.

    When the cheap metric is used to drive our quadruplet oracle and the
    expensive metric is the ground truth, this quantity is the empirical
    `φ` that Exp 1's setup section should report.

    Parameters
    ----------
    cheap_coords : (n, d_cheap) float array.
    expensive_coords : (n, d_expensive) float array. Must have the same n.
    n_samples : number of random quadruplets.
    seed : numpy RNG seed.
    """
    if cheap_coords.shape[0] != expensive_coords.shape[0]:
        raise ValueError(
            f"cheap has n={cheap_coords.shape[0]} but "
            f"expensive has n={expensive_coords.shape[0]}"
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    n = cheap_coords.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_samples, 4))
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]

    def _sq_diff(coords: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        diff = coords[u] - coords[v]
        return np.einsum("ij,ij->i", diff, diff)

    cheap_ab = _sq_diff(cheap_coords, a, b)
    cheap_cd = _sq_diff(cheap_coords, c, d)
    exp_ab = _sq_diff(expensive_coords, a, b)
    exp_cd = _sq_diff(expensive_coords, c, d)

    cheap_judgment = cheap_ab <= cheap_cd
    exp_judgment = exp_ab <= exp_cd

    return float(np.mean(cheap_judgment != exp_judgment))


def evaluate_coreset_coords_with_nmi(
        coreset_coords: np.ndarray,
        coreset_weights: Sequence[float],
        full_coords: np.ndarray,
        full_labels: Sequence[int],
        k: int,
        n_kmpp_seeds: int = 5,
        base_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run weighted k-means++ on the coreset and score each candidate two ways.

    For each of ``n_kmpp_seeds`` independent k-means++ runs on the
    weighted coreset, compute:

      - the k-means cost of the resulting centers on the full dataset P
        (same number :func:`evaluate_coreset_coords` returns), and
      - the normalized mutual information between the hard-assignment
        clustering of P induced by those centers and the ground-truth
        ``full_labels``.

    Entries at index ``i`` correspond to the same center set — so the
    caller can pick a candidate by any criterion and read off both
    metrics for that candidate.

    Parameters
    ----------
    coreset_coords : (m, d) float array.
    coreset_weights : length-m non-negative weights.
    full_coords : (n, d) float array. Cost is evaluated on the whole of P.
    full_labels : length-n integer labels. Treated as ground truth for
                  NMI. No requirement that there are exactly k classes.
    k : number of centers.
    n_kmpp_seeds : number of independent k-means++ runs.
    base_seed : run ``i`` uses ``base_seed + i``. Two calls with the same
                ``base_seed`` return identical arrays.

    Returns
    -------
    (costs, nmis) : two ``(n_kmpp_seeds,)`` arrays, aligned by index.
    """
    coords = np.ascontiguousarray(coreset_coords, dtype=np.float64)
    weights = np.asarray(coreset_weights, dtype=np.float64)
    labels = np.asarray(full_labels)

    if coords.ndim != 2:
        raise ValueError(
            f"coreset_coords must be 2-D, got shape {coords.shape}"
        )
    if weights.ndim != 1:
        raise ValueError(
            f"coreset_weights must be 1-D, got shape {weights.shape}"
        )
    if coords.shape[0] != weights.shape[0]:
        raise ValueError(
            f"coreset_coords has m={coords.shape[0]} but "
            f"coreset_weights has m={weights.shape[0]}"
        )
    if coords.shape[1] != full_coords.shape[1]:
        raise ValueError(
            f"coreset_coords has d={coords.shape[1]} but "
            f"full_coords has d={full_coords.shape[1]}"
        )
    if labels.shape[0] != full_coords.shape[0]:
        raise ValueError(
            f"full_labels has length {labels.shape[0]} but "
            f"full_coords has n={full_coords.shape[0]}"
        )
    if n_kmpp_seeds < 1:
        raise ValueError(f"n_kmpp_seeds must be >= 1, got {n_kmpp_seeds}")

    costs = np.empty(n_kmpp_seeds, dtype=np.float64)
    nmis = np.empty(n_kmpp_seeds, dtype=np.float64)

    for i in range(n_kmpp_seeds):
        rng = np.random.default_rng(base_seed + i)
        centers = _kmeans_plusplus_weighted(coords, weights, k, rng)

        costs[i] = weighted_kmeans_cost(full_coords, centers)

        # Hard assignment of P to centers, then NMI vs ground truth.
        # Vectorized nearest-center: (n, k) squared-distance matrix.
        # Memory is fine for the n we care about (~20K max here).
        sq = ((full_coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignments = sq.argmin(axis=1)
        nmis[i] = normalized_mutual_info_score(labels, assignments)

    return costs, nmis