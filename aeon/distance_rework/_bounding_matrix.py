import numpy as np
from numba import njit
import math


#@njit(cache=True)
def create_bounding_matrix(x_size: int, y_size: int, window: float = None):
    """Create a bounding matrix for a elastic distance.

    Parameters
    ----------
    x_size: int
        Size of the first time series.
    y_size: int
        Size of the second time series.
    window: float, optional
        Window size as a percentage of the smallest time series.
        If None, the bounding matrix will be full.

    Returns
    -------
    np.ndarray
        Bounding matrix where values in bound are finite and values out of bounds are
        infinite.

    Examples
    --------
    >>> create_bounding_matrix(10, 10, window=0.5)
    array(
    [[ 1.  1.  1. inf inf inf inf inf inf inf]
     [ 1.  1.  1.  1. inf inf inf inf inf inf]
     [inf  1.  1.  1.  1. inf inf inf inf inf]
     [inf inf  1.  1.  1.  1. inf inf inf inf]
     [inf inf inf  1.  1.  1.  1. inf inf inf]
     [inf inf inf inf  1.  1.  1.  1. inf inf]
     [inf inf inf inf inf  1.  1.  1.  1. inf]
     [inf inf inf inf inf inf  1.  1.  1.  1.]
     [inf inf inf inf inf inf inf  1.  1.  1.]
     [inf inf inf inf inf inf inf inf  1.  1.]]
     )
    """
    if window is None or window >= 1:
        return np.zeros((x_size, y_size))
    return _sakoe_chiba_bounding(x_size, y_size, window)


#@njit(cache=True)
def _sakoe_chiba_bounding(x_size: int, y_size: int,
                          radius_percent: float) -> np.ndarray:
    one_percent = min(x_size, y_size) / 100
    radius = math.ceil(((radius_percent * one_percent) * 100))
    bounding_matrix = np.full((x_size, y_size), np.inf)

    smallest_size = min(x_size, y_size)
    largest_size = max(x_size, y_size)

    width = largest_size - smallest_size + radius
    for i in range(smallest_size):
        lower = max(0, i - radius)
        upper = min(largest_size, i + width)
        for j in range(lower, upper):
            bounding_matrix[j, i] = 1.

    return bounding_matrix
