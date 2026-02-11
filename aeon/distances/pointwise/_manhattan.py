"""Manhattan distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._distance_factory._distance_factory import (
    build_distance,
    build_pairwise_distance,
)


@njit(cache=True, fastmath=True, inline="always")
def _univariate_manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Manhattan distance for univariate 1D arrays."""
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        distance += abs(x[i] - y[i])
    return distance


@njit(cache=True, fastmath=True)
def _manhattan_distance_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Manhattan distance for 2D inputs (n_channels, n_timepoints)."""
    distance = 0.0
    min_channels = min(x.shape[0], y.shape[0])
    for c in range(min_channels):
        distance += _univariate_manhattan_distance(x[c], y[c])
    return distance


manhattan_distance = build_distance(
    core_distance=_manhattan_distance_2d,
    name="manhattan",
)

manhattan_pairwise_distance = build_pairwise_distance(
    core_distance=manhattan_distance,
    name="manhattan",
)
