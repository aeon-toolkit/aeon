"""Squared distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._distance_factory._distance_factory import (
    build_distance,
    build_pairwise_distance,
)


@njit(cache=True, fastmath=True, inline="always")
def _univariate_squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        diff = x[i] - y[i]
        distance += diff * diff
    return distance


@njit(cache=True, fastmath=True)
def _squared_distance_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Squared distance for 2D inputs (n_channels, n_timepoints)."""
    distance = 0.0
    min_channels = min(x.shape[0], y.shape[0])
    for c in range(min_channels):
        distance += _univariate_squared_distance(x[c], y[c])
    return distance


squared_distance = build_distance(
    core_distance=_squared_distance_2d,
    name="squared",
)

squared_pairwise_distance = build_pairwise_distance(
    core_distance=squared_distance,
    name="squared",
)
