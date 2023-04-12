import numpy as np
from numba import njit
from aeon.distance_rework._squared import squared_distance


@njit(cache=True, fastmath=True)
def euclidean_distance(x: np.ndarray, y: np.ndarray):
    """Compute the euclidean distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.

    Returns
    -------
    float
        Euclidean distance between x and y.
    """
    return np.sqrt(squared_distance(x, y))
