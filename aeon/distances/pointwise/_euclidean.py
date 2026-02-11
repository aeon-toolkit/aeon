"""Euclidean distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._distance_factory._distance_factory import (
    build_distance,
    build_pairwise_distance,
)
from aeon.distances.pointwise._squared import (
    _squared_distance_2d,
    _univariate_squared_distance,
)


@njit(cache=True, fastmath=True, inline="always")
def _univariate_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance for univariate 1D arrays."""
    return np.sqrt(_univariate_squared_distance(x, y))


@njit(cache=True, fastmath=True)
def _euclidean_distance_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance for 2D inputs (n_channels, n_timepoints)."""
    return np.sqrt(_squared_distance_2d(x, y))


euclidean_distance = build_distance(
    core_distance=_euclidean_distance_2d,
    name="euclidean",
)

euclidean_pairwise_distance = build_pairwise_distance(
    core_distance=euclidean_distance,
    name="euclidean",
)
