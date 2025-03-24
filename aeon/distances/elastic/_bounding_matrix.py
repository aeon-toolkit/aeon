__maintainer__ = []

import math
from typing import Optional

import numpy as np
from numba import njit


@njit(cache=True)
def create_bounding_matrix(
    x_size: int,
    y_size: int,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
):
    """Create a bounding matrix for an elastic distance.

    Parameters
    ----------
    x_size : int
        Size of the first time series.
    y_size : int
        Size of the second time series.
    window : float, default=None
        Window size as a percentage of the smallest time series.
        If None, the bounding matrix will be full.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
        Itakura parallelogram does not support unequal length time series.

    Returns
    -------
    np.ndarray of shape (x_size, y_size)
        Bounding matrix where values in bound are True and values out of bounds are
        False.

    Examples
    --------
    >>> create_bounding_matrix(8, 8, window=0.5)
    array([[ True,  True,  True,  True,  True, False, False, False],
           [ True,  True,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True],
           [False,  True,  True,  True,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True,  True,  True],
           [False, False, False,  True,  True,  True,  True,  True]])
    """
    if itakura_max_slope is not None:
        if itakura_max_slope < 0 or itakura_max_slope > 1:
            raise ValueError("itakura_max_slope must be between 0 and 1")
        return _itakura_parallelogram(x_size, y_size, itakura_max_slope)
    if window is not None and window != 1.0:
        if window < 0 or window > 1:
            raise ValueError("window must be between 0 and 1")
        return _sakoe_chiba_bounding(x_size, y_size, window)
    return np.full((x_size, y_size), True)


@njit(cache=True)
def _itakura_parallelogram(x_size: int, y_size: int, max_slope_percent: float):
    """Itakura parallelogram bounding matrix.

    This code was adapted from pyts. This link to the original code:
    https://pyts.readthedocs.io/en/latest/_modules/pyts/metrics/dtw.html#itakura_parallelogram
    """
    one_percent = min(x_size, y_size) / 100
    max_slope = math.floor((max_slope_percent * one_percent) * 100)
    min_slope = 1 / float(max_slope)
    max_slope *= float(y_size - 1) / float(x_size - 2)
    max_slope = max(max_slope, 1.0)

    min_slope *= float(y_size - 2) / float(x_size - 1)
    min_slope = min(min_slope, 1.0)

    centered_scale = np.arange(x_size) - x_size + 1

    lower_bound = np.empty(x_size, dtype=np.float64)
    upper_bound = np.empty(x_size, dtype=np.float64)

    for i in range(x_size):
        lb0 = min_slope * i
        lb1 = max_slope * centered_scale[i] + y_size - 1
        lower_bound[i] = math.ceil(max(round(lb0, 2), round(lb1, 2)))

        ub0 = max_slope * i + 1
        ub1 = min_slope * centered_scale[i] + y_size
        upper_bound[i] = math.floor(min(round(ub0, 2), round(ub1, 2)))

    if max_slope == 1.0:
        if y_size > x_size:
            for i in range(x_size - 1):
                upper_bound[i] = lower_bound[i + 1]
        else:
            for i in range(x_size):
                upper_bound[i] = lower_bound[i] + 1

    for i in range(x_size):
        if lower_bound[i] < 0:
            lower_bound[i] = 0
        if lower_bound[i] > y_size:
            lower_bound[i] = y_size
        if upper_bound[i] < 0:
            upper_bound[i] = 0
        if upper_bound[i] > y_size:
            upper_bound[i] = y_size

    bounding_matrix = np.empty((x_size, y_size), dtype=np.bool_)
    for i in range(x_size):
        for j in range(y_size):
            bounding_matrix[i, j] = False

    for i in range(x_size):
        start = int(lower_bound[i])
        end = int(upper_bound[i])
        for j in range(start, end):
            bounding_matrix[i, j] = True

    return bounding_matrix


@njit(cache=True)
def _sakoe_chiba_bounding(
    x_size: int, y_size: int, radius_percent: float
) -> np.ndarray:

    if x_size > y_size:
        return _sakoe_chiba_bounding(y_size, x_size, radius_percent).T

    matrix = np.full((x_size, y_size), False)  # Create a matrix filled with False

    max_size = max(x_size, y_size) + 1

    shortest_dimension = min(x_size, y_size)
    thickness = int(radius_percent * shortest_dimension)
    for step in range(max_size):
        x_index = math.floor((step / max_size) * x_size)
        y_index = math.floor((step / max_size) * y_size)

        upper = max(0, (x_index - thickness))
        lower = min(x_size, (x_index + thickness + 1))

        matrix[upper:lower, y_index] = True

    return matrix
