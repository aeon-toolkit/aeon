from typing import List, Tuple
import numpy as np
from numba import njit
from aeon.distance_rework._dtw import (
    dtw_distance, dtw_cost_matrix, create_bounding_matrix, _dtw_distance,
    _dtw_cost_matrix
)
from aeon.distance_rework._alignment_paths import compute_min_return_path


@njit(fastmath=True, cache=True)
def ddtw_distance(x: np.ndarray, y: np.ndarray, window: float = None) -> float:
    """Compute the ddtw distance between two time series.

    DDTW is an adaptation of DTW originally proposed in [1]_. DDTW attempts to
    improve on dtw by better account for the 'shape' of the time series.
    This is done by considering y axis data points as higher level features of 'shape'.
    To do this the first derivative of the sequence is taken, and then using this
    derived sequence a dtw computation is done.

    The default derivative used is:

    .. math::
        D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Where q is the original time series and d_q is the derived time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
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

    References
    ----------
    .. [1] Keogh, Eamonn & Pazzani, Michael. (2002). Derivative Dynamic Time Warping.
        First SIAM International Conference on Data Mining.
        1. 10.1137/1.9781611972719.1.
    """
    x = average_of_slope(x)
    y = average_of_slope(y)
    return dtw_distance(x, y, window)


@njit(fastmath=True, cache=True)
def ddtw_cost_matrix(x: np.ndarray, y: np.ndarray, window: float = None) -> np.ndarray:
    """Compute the ddtw cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
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
    q: np.ndarray (n_channels, n_timepoints)
        Time series to take derivative of.

    Returns
    -------
    np.ndarray  (n_channels, n_timepoints - 2)
        Array containing the derivative of q.

    Raises
    ------
    ValueError
        If the time series has less than 3 points.

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
def ddtw_pairwise_distance(
        X: np.ndarray, window: float = None
) -> np.ndarray:
    """Compute the ddtw pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        ddtw pairwise matrix between the instances of X.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import ddtw_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> ddtw_pairwise_distance(X)
    array([[0.    , 1.    , 3.0625],
           [1.    , 0.    , 0.5625],
           [3.0625, 0.5625, 0.    ]])
    """
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
def ddtw_from_single_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None
) -> np.ndarray:
    """Compute the ddtw distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        Single time series.
    y: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_instances)
        ddtw distance between the collection of instances in y and the time series x.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import ddtw_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> ddtw_from_single_to_multiple_distance(x, y)
    array([0.25  , 2.25  , 5.0625])
    """
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1] - 2, y.shape[2] - 2, window)

    x = average_of_slope(x)
    for i in range(n_instances):
        distances[i] = _dtw_distance(x, average_of_slope(y[i]), bounding_matrix)

    return distances


@njit(cache=True, fastmath=True)
def ddtw_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None
) -> np.ndarray:
    """Compute the ddtw distance between two sets of time series.

    If x and y are the same then you should use ddtw_pairwise_distance.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    y: np.ndarray (m_instances, n_channels, n_timepoints)
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        ddtw distance between two collections of time series, x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import ddtw_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3, 3]],[[4, 5, 6, 9]], [[7, 8, 9, 22]]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> ddtw_from_multiple_to_multiple_distance(x, y)
    array([[ 7.5625, 14.0625,  5.0625],
           [12.25  , 20.25  ,  9.    ],
           [36.    , 49.    , 30.25  ]])
    """
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


@njit(cache=True, fastmath=True)
def ddtw_alignment_path(
        x: np.ndarray, y: np.ndarray, window: float = None
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the ddtw alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The ddtw distance betweeen the two time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import ddtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> ddtw_alignment_path(x, y)
    ([(0, 0), (1, 1)], 0.25)
    """
    bounding_matrix = create_bounding_matrix(x.shape[1] - 2, y.shape[1] - 2, window)
    cost_matrix = _dtw_cost_matrix(
        average_of_slope(x), average_of_slope(y), bounding_matrix
    )
    return compute_min_return_path(cost_matrix), cost_matrix[-1, -1]
