__author__ = ["chrisholder"]

import math

import numpy as np
from numba import njit


@njit(cache=True)
def create_bounding_matrix(
    x_size: int, y_size: int, window: float = None, itakura_max_slope: float = None
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
    if window is not None:
        if window < 0 or window > 1:
            raise ValueError("window must be between 0 and 1")
        return _sakoe_chiba_bounding(x_size, y_size, window)
    return np.full((x_size, y_size), True)


@njit(cache=True)
def _itakura_parallelogram(x_size: int, y_size: int, max_slope_percent: float):
    """Itakura parallelogram bounding matrix.

    This code was adapted from tslearn. This link to the orginal code line 974:
    https://github.com/tslearn-team/tslearn/blob/main/tslearn/metrics/dtw_variants.py
    """
    one_percent = min(x_size, y_size) / 100
    max_slope = math.floor((max_slope_percent * one_percent) * 100)
    min_slope = 1 / float(max_slope)
    max_slope *= float(x_size) / float(y_size)
    min_slope *= float(x_size) / float(y_size)

    lower_bound = np.empty((2, y_size))
    lower_bound[0] = min_slope * np.arange(y_size)
    lower_bound[1] = (
        (x_size - 1) - max_slope * (y_size - 1) + max_slope * np.arange(y_size)
    )
    lower_bound_ = np.empty(y_size)
    for i in range(y_size):
        lower_bound_[i] = max(round(lower_bound[0, i], 2), round(lower_bound[1, i], 2))
    lower_bound_ = np.ceil(lower_bound_)

    upper_bound = np.empty((2, y_size))
    upper_bound[0] = max_slope * np.arange(y_size)
    upper_bound[1] = (
        (x_size - 1) - min_slope * (y_size - 1) + min_slope * np.arange(y_size)
    )
    upper_bound_ = np.empty(y_size)
    for i in range(y_size):
        upper_bound_[i] = min(round(upper_bound[0, i], 2), round(upper_bound[1, i], 2))
    upper_bound_ = np.floor(upper_bound_ + 1)

    bounding_matrix = np.full((x_size, y_size), False)
    for i in range(y_size):
        bounding_matrix[int(lower_bound_[i]) : int(upper_bound_[i]), i] = True
    return bounding_matrix


@njit(cache=True)
def _sakoe_chiba_bounding(
    x_size: int, y_size: int, radius_percent: float
) -> np.ndarray:
    one_percent = min(x_size, y_size) / 100
    radius = math.floor((radius_percent * one_percent) * 100)
    bounding_matrix = np.full((x_size, y_size), False)

    smallest_size = min(x_size, y_size)
    largest_size = max(x_size, y_size)

    width = largest_size - smallest_size + radius
    for i in range(smallest_size):
        lower = max(0, i - radius)
        upper = min(largest_size, i + width) + 1
        bounding_matrix[i, lower:upper] = True

    return bounding_matrix
