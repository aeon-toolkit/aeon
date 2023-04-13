import numpy as np
from numba import njit
from aeon.distance_rework._wdtw import wdtw_distance, wdtw_cost_matrix, _wdtw_distance
from aeon.distance_rework._ddtw import average_of_slope
from aeon.distance_rework._bounding_matrix import create_bounding_matrix


@njit(cache=True, fastmath=True)
def wddtw_distance(x: np.ndarray, y: np.ndarray, window=None, g=0.05):
    """Compute the wddtw distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    float
        wddtw distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import wddtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wddtw_distance(x, y)
    0.0
    """
    x = average_of_slope(x)
    y = average_of_slope(y)
    return wdtw_distance(x, y, window, g)


@njit(cache=True, fastmath=True)
def wddtw_cost_matrix(x: np.ndarray, y: np.ndarray, window=None, g=0.05):
    """Compute the wddtw cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        wddtw cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import wddtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wddtw_cost_matrix(x, y)
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
    return wdtw_cost_matrix(x, y, window, g)


@njit(cache=True, fastmath=True)
def wddtw_pairwise_distance(X: np.ndarray, window=None, g: float = 0.05) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2] - 2, X.shape[2] - 2, window)

    X_average_of_slope = np.zeros((n_instances, X.shape[1], X.shape[2] - 2))
    for i in range(n_instances):
        X_average_of_slope[i] = average_of_slope(X[i])

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _wdtw_distance(
                X_average_of_slope[i], X_average_of_slope[j], bounding_matrix, g
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def wddtw_from_single_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window=None, g: float = 0.05
):
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1] - 2, y.shape[2] - 2, window)

    x = average_of_slope(x)
    for i in range(n_instances):
        distances[i] = _wdtw_distance(x, average_of_slope(y[i]), bounding_matrix, g)

    return distances


@njit(cache=True, fastmath=True)
def wddtw_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window=None, g: float = 0.05
):
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
            distances[i, j] = _wdtw_distance(
                derive_x[i], derive_y[j], bounding_matrix, g
            )
    return distances
