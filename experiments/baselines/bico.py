"""BICO baseline (Fichtenberger et al. 2013).

Thin wrapper around the ``algo-hhu/bico`` PyPI package, which vendors
the original Dortmund C++ implementation behind a sklearn-compatible
Python API.

No prepare/build split
----------------------
Unlike ``ours`` and ``sensitivity``, BICO's output depends directly on
the requested summary size: a different ``summary_size`` produces a
different streaming tree, with no shareable precomputation across the
sweep. So BICO is a single-call baseline — one ``build`` per sweep
point, no ``prepare``.

Output size is an upper bound
-----------------------------
BICO's streaming construction returns **up to** ``summary_size`` points,
typically fewer. Example at n=2000:

    summary_size=50   -> ~49
    summary_size=200  -> ~155
    summary_size=500  -> ~244
    summary_size=1000 -> ~229
    summary_size=5000 -> 2000  (capped at n)

This is intrinsic to BICO and not configurable. Wrappers in the
literature (sheikhomar, Draganov) accept it and plot against actual
output size; we do the same. Subsampling BICO's output to hit a fixed
target m would require duplicating rows past the natural tree size,
which adds no information.

Distance query accounting
-------------------------
BICO computes distances internally in C++; these do not flow through
``coreset_rm.oracles.DistanceOracle`` and cannot be counted. We emit
the sentinel ``-9999`` in the ``distance_queries`` field so downstream
code can distinguish "BICO's distance usage" (unknown, not measured)
from "zero distance queries"

Synthesized coreset points
--------------------------
BICO's summary points are tree centroids, not input points. In tests,
~80% of ``coreset_points_`` at summary_size=200 are not rows of the
input. Downstream evaluation must use ``evaluate_coreset_coords`` and
``evaluate_strong_coreset_coords`` (coordinate-based variants).

Dependency note
---------------
``bico`` pins ``numpy<2.0`` and will downgrade numpy 2.x on install.
Use a dedicated environment if other parts of the project require
numpy >= 2.
"""
from __future__ import annotations
import logging
import time

import numpy as np
from bico import BICO

from experiments.baselines.base import BaselineResult
from coreset_rm.rng import RNG

logger = logging.getLogger(__name__)

# Sentinel used in ``distance_queries`` to mark "distance usage
# present but not measurable through our oracle abstraction". Keeps
# downstream code (plots, tables) from confusing BICO's real
# distance work with a genuine zero (uniform, alg_g_as_coreset).
_DISTANCE_QUERIES_UNKNOWN = -9999


def build(
    coords: np.ndarray,
    k: int,
    m: int,
    root_rng: RNG,
) -> BaselineResult:
    """Build a BICO coreset on ``coords``.

    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        Input points.
    k : int
        Number of clusters — passed to BICO's ``n_clusters``.
    m : int
        Target (upper-bound) coreset size — passed as ``summary_size``.
        Actual output size may be smaller; read it from
        ``result.coreset_coords.shape[0]``.
    root_rng : RNG
        Root RNG. A named child is spawned to draw BICO's integer
        ``random_state`` so that repeated calls with the same (seed, m)
        are reproducible and different m values use independent seeds.

    Returns
    -------
    BaselineResult
        ``coreset_coords`` of shape (actual_m, d), where
        actual_m <= m; typically synthesized (not rows of ``coords``).
        ``coreset_weights`` sum to n. ``distance_queries = -9999``
        (see module docstring). ``quadruplet_queries = 0``.
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    # BICO wants a plain integer for random_state. Spawn per-m so each
    # sweep point draws independently and the same (root seed, m) pair
    # is reproducible.
    rng = root_rng.spawn(f"bico_m{m}").generator()
    bico_seed = int(rng.integers(0, 2**31 - 1))

    t0 = time.perf_counter()
    bico = BICO(
        n_clusters=k,
        summary_size=m,
        random_state=bico_seed,
        fit_coreset=True
    )
    bico.fit(coords)
    elapsed = time.perf_counter() - t0

    out_coords = np.ascontiguousarray(bico.coreset_points_, dtype=np.float64)
    out_weights = np.asarray(bico.coreset_weights_, dtype=np.float64)
    actual_m = int(out_coords.shape[0])

    logger.info(
        "bico.build n=%d k=%d requested_m=%d actual_m=%d time=%.3fs "
        "weights_sum=%.1f",
        coords.shape[0], k, m, actual_m, elapsed, out_weights.sum(),
    )
    if actual_m < m:
        logger.info(
            "bico.build actual_m=%d < requested_m=%d (BICO returns <= summary_size)",
            actual_m, m,
        )

    return BaselineResult(
        coreset_coords=out_coords,
        coreset_weights=out_weights,
        construction_time_s=elapsed,
        distance_queries=_DISTANCE_QUERIES_UNKNOWN,
        quadruplet_queries=0,
    )