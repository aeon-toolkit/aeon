import numpy as np
from numba import njit
from aeon.distance_rework._dtw import (
    dtw_distance, dtw_cost_matrix, create_bounding_matrix, _dtw_distance
)


@njit(fastmath=True, cache=True)
def ddtw_distance(x: np.ndarray, y: np.ndarray, window=None):
    """Compute the ddtw distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    Returns
    -------
    float
        ddtw distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import ddtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> ddtw_distance(x, y)
    0.0
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
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        ddtw cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import ddtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> ddtw_cost_matrix(x, y)
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    x = average_of_slope(x)
    y = average_of_slope(y)
    return dtw_cost_matrix(x, y, window)


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

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework._ddtw import average_of_slope
    >>> q = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> average_of_slope(q)
    array([[1., 1., 1., 1., 1., 1., 1., 1.]])
    """
    if q.shape[1] < 3:
        raise ValueError("Time series must have at least 3 points.")
    result = np.zeros((q.shape[0], q.shape[1] - 2))
    for i in range(q.shape[0]):
        for j in range(1, q.shape[1] - 1):
            result[i, j - 1] = ((q[i, j] - q[i, j - 1])
                                + (q[i, j + 1] - q[i, j - 1]) / 2.) / 2.
    return result

@njit(cache=True, fastmath=True)
def ddtw_pairwise_distance(X: np.ndarray, window=None) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2] - 2, X.shape[2] - 2, window)

    X_average_of_slope = np.zeros((n_instances, X.shape[1], X.shape[2] - 2))
    for i in range(n_instances):
        X_average_of_slope[i] = average_of_slope(X[i])

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _dtw_distance(
                X_average_of_slope[i], X_average_of_slope[j], bounding_matrix
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def ddtw_from_single_to_multiple_distance(x: np.ndarray, y: np.ndarray, window=None):
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1] - 2, y.shape[2] - 2, window)

    x = average_of_slope(x)
    for i in range(n_instances):
        distances[i] = _dtw_distance(x, average_of_slope(y[i]), bounding_matrix)

    return distances


@njit(cache=True, fastmath=True)
def ddtw_from_multiple_to_multiple_distance(x: np.ndarray, y: np.ndarray, window=None):
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    # Derive the arrays before so that we dont have to redo every iteration
    derive_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] - 2))
    for i in range(x.shape[0]):
        derive_x[i] = average_of_slope(x[i])

    derive_y = np.zeros((y.shape[0], y.shape[1], y.shape[2] - 2))
    for i in range(y.shape[0]):
        derive_y[i] = average_of_slope(y[i])

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _dtw_distance(
                derive_x[i], derive_y[j], bounding_matrix
            )
    return distances
