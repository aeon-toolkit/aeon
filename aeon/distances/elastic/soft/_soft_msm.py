r"""Soft move-split-merge (soft-MSM) distance between two time series."""

__maintainer__ = []
from typing import Optional, Union

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic._msm import _cost_independent
from aeon.distances.elastic.soft._soft_distance_utils import (
    _compute_soft_gradient,
    _soft_min,
    _soft_min_with_arrs,
)
from aeon.utils._threading import threaded
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate

MAX_NP_FLOAT = np.finfo(np.float64).max


@njit(cache=True, fastmath=True)
def soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    c: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(_x, _y, bounding_matrix, c, gamma)
    elif x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(x, y, bounding_matrix, c, gamma)
    else:
        raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    c: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_cost_matrix(_x, _y, bounding_matrix, c, gamma)
    elif x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_cost_matrix(x, y, bounding_matrix, c, gamma)
    else:
        raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
) -> float:
    cm = _soft_msm_cost_matrix(x, y, bounding_matrix, c, gamma)
    return abs(cm[x.shape[1] - 1, y.shape[1] - 1])


@njit(cache=True, fastmath=True)
def _soft_msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))

    for ch in range(x.shape[0]):
        cost_matrix_per_ch = _soft_msm_univariate_cost_matrix(
            x[ch], y[ch], bounding_matrix, c, gamma
        )
        cost_matrix += cost_matrix_per_ch

    return cost_matrix


@njit(cache=True, fastmath=True)
def _soft_msm_univariate_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float, gamma: float
) -> np.ndarray:
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = np.abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            split = cost_matrix[i - 1, 0] + _cost_independent(x[i], x[i - 1], y[0], c)
            cost_matrix[i, 0] = _soft_min(MAX_NP_FLOAT, split, MAX_NP_FLOAT, gamma)

    for i in range(1, y_size):
        if bounding_matrix[0, i]:
            merge = cost_matrix[0, i - 1] + _cost_independent(y[i], x[0], y[i - 1], c)
            cost_matrix[0, i] = _soft_min(MAX_NP_FLOAT, MAX_NP_FLOAT, merge, gamma)

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                move = cost_matrix[i - 1][j - 1] + np.abs(x[i] - y[j])
                split = cost_matrix[i - 1][j] + _cost_independent(
                    x[i], x[i - 1], y[j], c
                )
                merge = cost_matrix[i][j - 1] + _cost_independent(
                    y[j], x[i], y[j - 1], c
                )
                cost_matrix[i, j] = _soft_min(move, split, merge, gamma)

    return cost_matrix


@threaded
def soft_msm_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    gamma: float = 1.0,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _soft_msm_pairwise_distance(
            _X, window, c, itakura_max_slope, gamma, unequal_length
        )

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_msm_from_multiple_to_multiple_distance(
        _X, _y, window, c, itakura_max_slope, gamma, unequal_length
    )


@njit(cache=True, fastmath=True, parallel=True)
def _soft_msm_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    c: float,
    itakura_max_slope: Optional[float],
    gamma: float,
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in prange(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_msm_distance(x1, x2, bounding_matrix, c, gamma)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _soft_msm_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    c: float,
    itakura_max_slope: Optional[float],
    gamma: float,
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in prange(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_msm_distance(x1, y1, bounding_matrix, c, gamma)
    return distances


@njit(cache=True, fastmath=True)
def soft_msm_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    gamma: float = 1.0,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    cm = soft_msm_cost_matrix(x, y, window, c, gamma, itakura_max_slope)
    return compute_min_return_path(cm), abs(cm[x.shape[-1] - 1, y.shape[-1] - 1])


@njit(cache=True, fastmath=True)
def _soft_msm_cost_matrix_with_arr_independent(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size, y_size))
    move_arr = np.zeros((x_size, y_size))
    split_arr = np.zeros((x_size, y_size))
    merge_arr = np.zeros((x_size, y_size))
    diff_dist_matrix = np.zeros((x_size, y_size))

    for ch in range(x.shape[0]):
        ind_cost_matrix = np.zeros((x_size, y_size))
        ind_move_arr = np.zeros((x_size, y_size))
        ind_split_arr = np.zeros((x_size, y_size))
        ind_merge_arr = np.zeros((x_size, y_size))
        ind_diff_dist_matrix = np.zeros((x_size, y_size))

        _soft_msm_univariate_cost_matrix_with_arr(
            x[ch],
            y[ch],
            bounding_matrix,
            gamma,
            c,
            ind_cost_matrix,
            ind_move_arr,
            ind_split_arr,
            ind_merge_arr,
            ind_diff_dist_matrix,
        )

        cost_matrix = np.add(cost_matrix, ind_cost_matrix)
        move_arr = np.add(move_arr, ind_move_arr)
        split_arr = np.add(split_arr, ind_split_arr)
        merge_arr = np.add(merge_arr, ind_merge_arr)
        diff_dist_matrix = np.add(diff_dist_matrix, ind_diff_dist_matrix)

    return cost_matrix, move_arr, split_arr, merge_arr, diff_dist_matrix


@njit(cache=True, fastmath=True)
def _cost_independent_with_diff(x: float, y: float, z: float, c: float) -> float:
    if (y <= x <= z) or (y >= x >= z):
        return c
    return c + min(abs(x - y), abs(x - z))


@njit(cache=True, fastmath=True)
def _soft_msm_univariate_cost_matrix_with_arr(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    c: float,
    cost_matrix: np.ndarray,
    move_arr: np.ndarray,
    split_arr: np.ndarray,
    merge_arr: np.ndarray,
    diff_dist_matrix: np.ndarray,
):
    """Compute soft msm cost matrix and arrays for univariate time series.

    This method MUTATES: cost_matrix, move_arr, split_arr, merge_arr.

    This isn't intended for public consumption so I decided it's probably ok to
    mutate these arrays in-place. This is a performance optimisation to avoid
    unnecessary memory allocations.
    """
    x_size = x.shape[0]
    y_size = y.shape[0]

    cost_matrix[0, 0] = np.abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            split = cost_matrix[i - 1, 0] + _cost_independent(x[i], x[i - 1], y[0], c)
            diff_dist_matrix[i, 0] += x[i] - y[0]
            cost_matrix[i, 0] = _soft_min_with_arrs(
                MAX_NP_FLOAT,
                split,
                MAX_NP_FLOAT,
                gamma,
                move_arr,
                split_arr,
                merge_arr,
                i,
                0,
            )

    for j in range(1, y_size):
        if bounding_matrix[0, j]:
            merge = cost_matrix[0, j - 1] + _cost_independent(y[j], x[0], y[j - 1], c)
            diff_dist_matrix[0, j] += x[0] - y[j]
            cost_matrix[0, j] = _soft_min_with_arrs(
                MAX_NP_FLOAT,
                MAX_NP_FLOAT,
                merge,
                gamma,
                move_arr,
                split_arr,
                merge_arr,
                0,
                j,
            )

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                diff = x[i] - y[j]
                move = cost_matrix[i - 1, j - 1] + np.abs(diff)
                diff_dist_matrix[i, j] += diff
                split = cost_matrix[i - 1, j] + _cost_independent(
                    x[i], x[i - 1], y[j], c
                )
                merge = cost_matrix[i, j - 1] + _cost_independent(
                    y[j], x[i], y[j - 1], c
                )
                cost_matrix[i, j] = cost_matrix[i, j] = _soft_min_with_arrs(
                    move, split, merge, gamma, move_arr, split_arr, merge_arr, i, j
                )


def soft_msm_gradient(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    return _compute_soft_gradient(
        x,
        y,
        _soft_msm_cost_matrix_with_arr_independent,
        gamma=gamma,
        window=window,
        itakura_max_slope=itakura_max_slope,
        c=c,
    )
