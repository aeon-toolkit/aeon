import numpy as np
from numba import njit
from aeon.distance_rework._squared import univariate_squared_distance
from aeon.distance_rework._bounding_matrix import create_bounding_matrix


@njit(cache=True, fastmath=True)
def dtw_distance(x: np.ndarray, y: np.ndarray, window=None) -> float:
    """Compute the dtw distance between two time series.

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
        dtw distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import dtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> dtw_distance(x, y)
    0.0
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _dtw_distance(x, y, bounding_matrix)


@njit(cache=True, fastmath=True)
def dtw_cost_matrix(x: np.ndarray, y: np.ndarray, window=None) -> np.ndarray:
    """Compute the dtw cost matrix between two time series.

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
        dtw cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import dtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> dtw_cost_matrix(x, y)
    array([[  0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204., 285.],
           [  1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204.],
           [  5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140.],
           [ 14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91.],
           [ 30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.],
           [ 55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.],
           [ 91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.],
           [140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.],
           [204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.],
           [285., 204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.]])
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _dtw_cost_matrix(x, y, bounding_matrix)


@njit(cache=True, fastmath=True)
def _dtw_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray
) -> float:
    return _dtw_cost_matrix(x, y, bounding_matrix)[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _dtw_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = \
                    univariate_squared_distance(x[:, i], y[:, j]) + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]
