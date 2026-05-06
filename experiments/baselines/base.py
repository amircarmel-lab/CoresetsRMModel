"""Common return type for baseline coreset construction algorithms."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    """Return type of every baseline's ``build`` function.

    Attributes
    ----------
    coreset_coords : np.ndarray, shape (m, d)
        The coreset points in coordinate space. May or may not be input
        points — wrappers do not distinguish.
    coreset_weights : np.ndarray, shape (m,)
        Non-negative weights, one per coreset point.
    construction_time_s : float
        Wall-clock time spent inside the construction call, in seconds.
    distance_queries : int
        Number of exact distance queries made during construction.
    quadruplet_queries : int
        Number of quadruplet-oracle queries made during construction.
    """

    coreset_coords: np.ndarray
    coreset_weights: np.ndarray
    construction_time_s: float
    distance_queries: int
    quadruplet_queries: int