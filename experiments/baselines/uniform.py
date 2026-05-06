"""Uniform sampling baseline."""
from __future__ import annotations
import logging
import time

import numpy as np

from documentation import BaselineResult
from coreset_rm.rng import RNG

logger = logging.getLogger(__name__)


def build(
    coords: np.ndarray,
    m: int,
    root_rng: RNG,
) -> BaselineResult:
    """Construct a uniform-sampling coreset of size ``m``."""
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    n = coords.shape[0]
    if n == 0:
        raise ValueError("coords is empty")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    rng = root_rng.spawn("uniform").generator()

    t0 = time.perf_counter()

    sampled_positions = rng.choice(n, size=m, replace=True)
    sampled_coords = coords[sampled_positions]
    sampled_weights = np.full(m, n / m, dtype=np.float64)

    elapsed = time.perf_counter() - t0

    logger.info("uniform.build n=%d m=%d time=%.3fs", n, m, elapsed)

    return BaselineResult(
        coreset_coords=sampled_coords,
        coreset_weights=sampled_weights,
        construction_time_s=elapsed,
        distance_queries=0,
        quadruplet_queries=0,
    )