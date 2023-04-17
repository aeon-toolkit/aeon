# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._squared import univariate_squared_distance


@njit
def compute_min_return_path(cost_matrix: np.ndarray) -> List[Tuple]:
    """Compute the minimum return path through a cost matrix.

    Parameters
    ----------
    cost_matrix: np.ndarray (n_timepoints_x, n_timepoints_y)
        Cost matrix.

    Returns
    -------
    List[Tuple]
        List of indices that make up the minimum return path.
    """
    x_size, y_size = cost_matrix.shape
    i, j = x_size - 1, y_size - 1
    alignment = []

    while i > 0 or j > 0:
        alignment.append((i, j))

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_index = np.argmin(
                np.array(
                    [
                        cost_matrix[i - 1, j - 1],
                        cost_matrix[i - 1, j],
                        cost_matrix[i, j - 1],
                    ]
                )
            )

            if min_index == 0:
                i, j = i - 1, j - 1
            elif min_index == 1:
                i -= 1
            else:
                j -= 1

    alignment.append((0, 0))
    return alignment[::-1]


@njit
def compute_lcss_return_path(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    bounding_matrix: np.ndarray,
    cost_matrix: np.ndarray,
) -> List[Tuple]:
    """Compute the return path through a cost matrix for the LCSS algorithm.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints_x)
        First time series.
    y: np.ndarray (n_channels, n_timepoints_y)
        Second time series.
    epsilon: float
        Threshold for the LCSS algorithm.
    bounding_matrix: np.ndarray (n_timepoints_x, n_timepoints_y)
        Bounding matrix for the LCSS algorithm.
    cost_matrix: np.ndarray (n_timepoints_x, n_timepoints_y)
        Cost matrix for the LCSS algorithm.

    Returns
    -------
    List[Tuple]
        List of indices that make up the return path.
    """
    x_size, y_size = cost_matrix.shape

    i, j = (x_size, y_size)
    path = []

    while i > 0 and j > 0:
        if bounding_matrix[i - 1, j - 1]:
            if univariate_squared_distance(x[:, i - 1], y[:, j - 1]) <= epsilon:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
            elif cost_matrix[i - 1][j] > cost_matrix[i][j - 1]:
                i = i - 1
            else:
                j = j - 1
    return path[::-1]


@njit(cache=True)
def _add_inf_to_out_of_bounds_cost_matrix(
    cost_matrix: np.ndarray, bounding_matrix: np.ndarray
) -> np.ndarray:
    x_size, y_size = cost_matrix.shape
    for i in range(x_size):
        for j in range(y_size):
            if not np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i, j] = np.inf

    return cost_matrix
