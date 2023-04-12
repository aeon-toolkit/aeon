import numpy as np
from numba import njit
from aeon.distance_rework._wdtw import wdtw_distance, wdtw_cost_matrix
from aeon.distance_rework._ddtw import average_of_slope

@njit(fastmath=True, cache=True)
def wddtw_distance(x: np.ndarray, y: np.ndarray, window=None, g=0.05):
    """Compute the wddtw distance between two time series.

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


@njit(fastmath=True, cache=True)
def wddtw_cost_matrix(x: np.ndarray, y: np.ndarray, window=None, g=0.05):
    """Compute the wddtw cost matrix between two time series.

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
