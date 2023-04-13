import numpy as np
from numba import njit
from aeon.distance_rework._squared import univariate_squared_distance
from aeon.distance_rework._bounding_matrix import create_bounding_matrix


@njit(cache=True, fastmath=True)
def lcss_distance(x: np.ndarray, y: np.ndarray, window=None, epsilon: float = 1.) -> float:
    """Returns the lcss distance between x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series.
    y : np.ndarray
        Second time series.
    window : float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    float
        The lcss distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import lcss_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> lcss_distance(x, y)
    0.0
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _lcss_distance(x, y, bounding_matrix, epsilon)


@njit(cache=True, fastmath=True)
def lcss_cost_matrix(
        x: np.ndarray, y: np.ndarray, window=None, epsilon: float = 1.
) -> np.ndarray:
    """Returns the lcss cost matrix between x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series.
    y : np.ndarray
        Second time series.
    window : float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.

    Returns
    -------
    np.ndarray
        The lcss cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import lcss_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> lcss_cost_matrix(x, y)
    array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
           [ 1.,  2.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
           [ 1.,  2.,  3.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
           [ 1.,  2.,  3.,  4.,  5.,  5.,  5.,  5.,  5.,  5.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  6.,  6.,  6.,  6.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  7.,  7.,  7.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.,  8.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  9.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _lcss_cost_matrix(x, y, bounding_matrix, epsilon)


@njit(cache=True, fastmath=True)
def _lcss_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float = 1.
) -> float:
    distance = _lcss_cost_matrix(
        x, y, bounding_matrix, epsilon
    )[x.shape[1] - 1, y.shape[1] - 1]
    return 1 - float(distance / min(x.shape[1], y.shape[1]))


@njit(cache=True, fastmath=True)
def _lcss_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon: float = 1.
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                squared_distance = univariate_squared_distance(x[:, i - 1], y[:, j - 1])
                if squared_distance <= epsilon:
                    cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                else:
                    cost_matrix[i, j] = max(
                        cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                    )

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def lcss_pairwise_distance(
        X: np.ndarray, window: float = None, epsilon: float = 1.
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _lcss_distance(X[i], X[j], bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def lcss_from_single_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.
):
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[2], window)

    for i in range(n_instances):
        distances[i] = _lcss_distance(x, y[i], bounding_matrix, epsilon)

    return distances


@njit(cache=True, fastmath=True)
def lcss_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, epsilon: float = 1.
):
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _lcss_distance(x[i], y[j], bounding_matrix, epsilon)
    return distances
