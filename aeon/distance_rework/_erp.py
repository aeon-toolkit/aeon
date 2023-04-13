import numpy as np
from numba import njit
from aeon.distance_rework._squared import univariate_squared_distance
from aeon.distance_rework._bounding_matrix import create_bounding_matrix

@njit(cache=True, fastmath=True)
def erp_distance(x: np.ndarray, y: np.ndarray, window=None, g: float = 0.) -> float:
    """Compute the ERP distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.
        The reference value to penalise gaps. The default is 0.

    Returns
    -------
    float
        ERP distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import erp_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> erp_distance(x, y)
    0.0
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _erp_distance(x, y, bounding_matrix, g)


@njit(cache=True, fastmath=True)
def erp_cost_matrix(x: np.ndarray, y: np.ndarray, window=None, g: float = 0.) -> np.ndarray:
    """Compute the ERP cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.
        The reference value to penalise gaps. The default is 0.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        ERP cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import erp_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> erp_cost_matrix(x, y)
    array([[  0.,   4.,  13.,  29.,  54.,  90., 139., 203., 284., 384.],
           [  4.,   0.,   5.,  17.,  38.,  70., 115., 175., 252., 348.],
           [ 13.,   5.,   0.,   6.,  21.,  47.,  86., 140., 211., 301.],
           [ 29.,  17.,   6.,   0.,   7.,  25.,  56., 102., 165., 247.],
           [ 54.,  38.,  21.,   7.,   0.,   8.,  29.,  65., 118., 190.],
           [ 90.,  70.,  47.,  25.,   8.,   0.,   9.,  33.,  74., 134.],
           [139., 115.,  86.,  56.,  29.,   9.,   0.,  10.,  37.,  83.],
           [203., 175., 140., 102.,  65.,  33.,  10.,   0.,  11.,  41.],
           [284., 252., 211., 165., 118.,  74.,  37.,  11.,   0.,  12.],
           [384., 348., 301., 247., 190., 134.,  83.,  41.,  12.,   0.]])
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _erp_cost_matrix(x, y, bounding_matrix, g)


@njit(cache=True, fastmath=True)
def _erp_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float = 0.
) -> float:
    return _erp_cost_matrix(x, y, bounding_matrix, g)[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _erp_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float = 0.
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    gx_distance, x_sum = _precompute_g(x, g)
    gy_distance, y_sum = _precompute_g(y, g)

    cost_matrix[1:, 0] = x_sum
    cost_matrix[0, 1:] = y_sum

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] +
                    univariate_squared_distance(x[:, i - 1], y[:, j - 1]),
                    cost_matrix[i - 1, j] + gx_distance[i - 1],
                    cost_matrix[i, j - 1] + gy_distance[j - 1],
                )

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _precompute_g(x: np.ndarray, g: float):
    gx_distance = np.zeros(x.shape[1])
    g_arr = np.full(x.shape[0], g)
    x_sum = 0

    for i in range(x.shape[1]):
        temp = univariate_squared_distance(x[:, i], g_arr)
        gx_distance[i] = temp
        x_sum += temp
    return gx_distance, x_sum


@njit(cache=True, fastmath=True)
def erp_pairwise_distance(
        X: np.ndarray, window: float = None, g: float = 0.
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _erp_distance(X[i], X[j], bounding_matrix, g)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def erp_from_single_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.
):
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[2], window)

    for i in range(n_instances):
        distances[i] = _erp_distance(x, y[i], bounding_matrix, g)

    return distances


@njit(cache=True, fastmath=True)
def erp_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.
):
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _erp_distance(x[i], y[j], bounding_matrix, g)
    return distances
