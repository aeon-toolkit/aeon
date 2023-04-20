# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

import math

import numpy as np
from numba import njit


@njit(cache=True)
def create_bounding_matrix(x_size: int, y_size: int, window: float = None):
    """Create a bounding matrix for a elastic distance.

    Parameters
    ----------
    x_size: int
        Size of the first time series.
    y_size: int
        Size of the second time series.
    window: float, defaults=None
        Window size as a percentage of the smallest time series.
        If None, the bounding matrix will be full.

    Returns
    -------
    np.ndarray
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
    if window is None or window >= 1:
        return np.full((x_size, y_size), True)
    return _sakoe_chiba_bounding(x_size, y_size, window)


@njit(cache=True)
def _sakoe_chiba_bounding(
    x_size: int, y_size: int, radius_percent: float
) -> np.ndarray:
    one_percent = min(x_size, y_size) / 100
    radius = math.floor(((radius_percent * one_percent) * 100))
    bounding_matrix = np.full((x_size, y_size), False)

    smallest_size = min(x_size, y_size)
    largest_size = max(x_size, y_size)

    width = largest_size - smallest_size + radius
    for i in range(smallest_size):
        lower = max(0, i - radius)
        upper = min(largest_size, i + width) + 1
        bounding_matrix[i, lower:upper] = True

    return bounding_matrix
