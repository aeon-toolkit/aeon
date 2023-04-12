import numpy as np
from numba import njit
from aeon.distance_rework._utils import (
    pairwise_distance, distance_from_single_to_multiple,
    distance_from_multiple_to_multiple
)


@njit(cache=True, fastmath=True)
def squared_distance(x: np.ndarray, y: np.ndarray):
    """Compute the squared distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.

    Returns
    -------
    float
        Squared distance between x and y.
    """
    distance = 0.0
    for i in range(x.shape[0]):
        distance += univariate_squared_distance(x[i], y[i])
    return distance


@njit(cache=True, fastmath=True)
def univariate_squared_distance(x: np.ndarray, y: np.ndarray):
    """Compute the squared distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_timepoints)
        First time series.
    y: np.ndarray (n_timepoints)
        Second time series.

    Returns
    -------
    float
        Squared distance between x and y.
    """
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        difference = x[i] - y[i]
        distance += difference * difference
    return distance
