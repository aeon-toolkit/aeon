import numpy as np
from numba import njit
from aeon.distance_rework._dtw import dtw_distance, dtw_cost_matrix


@njit(fastmath=True, cache=True)
def average_of_slope(q: np.ndarray) -> np.ndarray:
    r"""Compute the average of a slope between points.
    Computes the average of the slope of the line through the point in question and
    its left neighbour, and the slope of the line through the left neighbour and the
    right neighbour. proposed in [1] for use in this context.
    .. math::
    q'_(i) = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}
    Where q is the original time series and q' is the derived time series.

    Parameters
    ----------
    q: np.ndarray (n_dims, n_timepoints)
        Time series to take derivative of.

    Returns
    -------
    np.ndarray  (n_dims, n_timepoints - 2)
        Array containing the derivative of q.
    """
    result = np.zeros((q.shape[0], q.shape[1] - 2))
    for i in range(q.shape[0]):
        for j in range(1, q.shape[1] - 1):
            result[i, j - 1] = ((q[i, j] - q[i, j - 1])
                                + (q[i, j + 1] - q[i, j - 1]) / 2.) / 2.
    return result


@njit(fastmath=True, cache=True)
def ddtw_distance(x: np.ndarray, y: np.ndarray, window=None):
    """Compute the ddtw distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, optional
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    Returns
    -------
    float
        ddtw distance between x and y.
    """
    x = average_of_slope(x)
    y = average_of_slope(y)
    return dtw_distance(x, y, window)


@njit(fastmath=True, cache=True)
def ddtw_cost_matrix(x: np.ndarray, y: np.ndarray, window=None):
    """Compute the ddtw cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, optional
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        ddtw cost matrix between x and y.
    """
    x = average_of_slope(x)
    y = average_of_slope(y)
    return dtw_cost_matrix(x, y, window)
