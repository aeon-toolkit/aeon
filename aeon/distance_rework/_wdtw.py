import numpy as np
from numba import njit
from aeon.distance_rework._squared import univariate_squared_distance
from aeon.distance_rework._bounding_matrix import create_bounding_matrix


@njit(cache=True, fastmath=True)
def wdtw_distance(x: np.ndarray, y: np.ndarray, window=None, g=0.05) -> float:
    """Compute the wdtw distance between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, optional
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, optional
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    float
        wdtw distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import wdtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wdtw_distance(x, y)
    0.0
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _wdtw_distance(x, y, bounding_matrix, g)


@njit(cache=True, fastmath=True)
def wdtw_cost_matrix(x: np.ndarray, y: np.ndarray, window=None, g=0.05) -> np.ndarray:
    """Compute the wdtw cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (n_dims, n_timepoints)
        First time series.
    y: np.ndarray (n_dims, n_timepoints)
        Second time series.
    window: float, optional
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, optional
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        wdtw cost matrix between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distance_rework import wdtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wdtw_cost_matrix(x, y)
    array([[  0.        ,   0.450166  ,   2.30044662,   6.57563393,
             14.37567559,  26.87567559,  45.32558186,  71.04956205,
            105.44507215, 149.98162593],
           [  0.450166  ,   0.        ,   0.450166  ,   2.30044662,
              6.57563393,  14.37567559,  26.87567559,  45.32558186,
             71.04956205, 105.44507215],
           [  2.30044662,   0.450166  ,   0.        ,   0.450166  ,
              2.30044662,   6.57563393,  14.37567559,  26.87567559,
             45.32558186,  71.04956205],
           [  6.57563393,   2.30044662,   0.450166  ,   0.        ,
              0.450166  ,   2.30044662,   6.57563393,  14.37567559,
             26.87567559,  45.32558186],
           [ 14.37567559,   6.57563393,   2.30044662,   0.450166  ,
              0.        ,   0.450166  ,   2.30044662,   6.57563393,
             14.37567559,  26.87567559],
           [ 26.87567559,  14.37567559,   6.57563393,   2.30044662,
              0.450166  ,   0.        ,   0.450166  ,   2.30044662,
              6.57563393,  14.37567559],
           [ 45.32558186,  26.87567559,  14.37567559,   6.57563393,
              2.30044662,   0.450166  ,   0.        ,   0.450166  ,
              2.30044662,   6.57563393],
           [ 71.04956205,  45.32558186,  26.87567559,  14.37567559,
              6.57563393,   2.30044662,   0.450166  ,   0.        ,
              0.450166  ,   2.30044662],
           [105.44507215,  71.04956205,  45.32558186,  26.87567559,
             14.37567559,   6.57563393,   2.30044662,   0.450166  ,
              0.        ,   0.450166  ],
           [149.98162593, 105.44507215,  71.04956205,  45.32558186,
             26.87567559,  14.37567559,   6.57563393,   2.30044662,
              0.450166  ,   0.        ]])
    """
    bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
    return _wdtw_cost_matrix(x, y, bounding_matrix, g)


@njit(cache=True, fastmath=True)
def _wdtw_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g=0.05
) -> float:
    return _wdtw_cost_matrix(x, y, bounding_matrix, g)[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _wdtw_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g=0.05
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    weight_vector = np.array(
        [1 / (1 + np.exp(-g * (i - x_size / 2))) for i in range(0, x_size)]
    )

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i + 1, j + 1] = \
                    univariate_squared_distance(x[:, i], y[:, j]) * weight_vector[
                    abs(i - j)
                ] + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]
