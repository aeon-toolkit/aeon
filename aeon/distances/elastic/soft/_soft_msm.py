r"""Soft move-split-merge (soft-MSM) distance between two time series."""

__maintainer__ = []
from typing import Optional

import numpy as np
from numba import njit

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_distance_utils import (
    _compute_soft_gradient,
    _softmin3,
)


@njit(cache=True, fastmath=True)
def soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
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
        return _soft_msm_distance(_x, _y, bounding_matrix, independent, c, gamma)
    elif x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(x, y, bounding_matrix, independent, c, gamma)
    else:
        raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
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
        if independent:
            return _soft_msm_independent_cost_matrix(_x, _y, bounding_matrix, c, gamma)
        else:
            return _soft_msm_dependent_cost_matrix(_x, _y, bounding_matrix, c, gamma)
    elif x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        if independent:
            return _soft_msm_independent_cost_matrix(x, y, bounding_matrix, c, gamma)
        else:
            return _soft_msm_dependent_cost_matrix(x, y, bounding_matrix, c, gamma)
    else:
        raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    independent: bool,
    c: float,
    gamma: float,
) -> float:
    if independent:
        cm = _soft_msm_independent_cost_matrix(x, y, bounding_matrix, c, gamma)
    else:
        cm = _soft_msm_dependent_cost_matrix(x, y, bounding_matrix, c, gamma)
    return cm[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _soft_msm_independent_cost_matrix(
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
    cost_matrix[0, 0] = abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost_matrix[i, 0] = cost_matrix[i - 1, 0] + _cost_independent(
                x[i], x[i - 1], y[0], c
            )

    for j in range(1, y_size):
        if bounding_matrix[0, j]:
            cost_matrix[0, j] = cost_matrix[0, j - 1] + _cost_independent(
                y[j], x[0], y[j - 1], c
            )

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                move_val = cost_matrix[i - 1, j - 1] + abs(x[i] - y[j])
                split_val = cost_matrix[i - 1, j] + _cost_independent(
                    x[i], x[i - 1], y[j], c
                )
                merge_val = cost_matrix[i, j - 1] + _cost_independent(
                    y[j], x[i], y[j - 1], c
                )
                cost_matrix[i, j] = _softmin3(move_val, split_val, merge_val, gamma)

    return cost_matrix


@njit(cache=True, fastmath=True)
def _soft_msm_dependent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float, gamma: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = np.sum(np.abs(x[:, 0] - y[:, 0]))

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost_matrix[i, 0] = cost_matrix[i - 1, 0] + _cost_dependent(
                x[:, i], x[:, i - 1], y[:, 0], c
            )

    for j in range(1, y_size):
        if bounding_matrix[0, j]:
            cost_matrix[0, j] = cost_matrix[0, j - 1] + _cost_dependent(
                y[:, j], x[:, 0], y[:, j - 1], c
            )

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                move_val = cost_matrix[i - 1, j - 1] + np.sum(np.abs(x[:, i] - y[:, j]))
                split_val = cost_matrix[i - 1, j] + _cost_dependent(
                    x[:, i], x[:, i - 1], y[:, j], c
                )
                merge_val = cost_matrix[i, j - 1] + _cost_dependent(
                    y[:, j], x[:, i], y[:, j - 1], c
                )
                cost_matrix[i, j] = _softmin3(move_val, split_val, merge_val, gamma)

    return cost_matrix


@njit(cache=True, fastmath=True)
def _cost_independent(x_val: float, y_val: float, z_val: float, c: float) -> float:
    if (y_val <= x_val <= z_val) or (y_val >= x_val >= z_val):
        return c
    else:
        return c + min(abs(x_val - y_val), abs(x_val - z_val))


@njit(cache=True, fastmath=True)
def _cost_dependent(x: np.ndarray, y: np.ndarray, z: np.ndarray, c: float) -> float:
    in_between = True
    for d in range(x.shape[0]):
        if not ((y[d] <= x[d] <= z[d]) or (y[d] >= x[d] >= z[d])):
            in_between = False
            break
    if in_between:
        return c
    else:
        dist_xy = np.sum(np.abs(x - y))
        dist_xz = np.sum(np.abs(x - z))
        return c + min(dist_xy, dist_xz)


@njit(cache=True, fastmath=True)
def soft_msm_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    independent: bool = True,
    c: float = 1.0,
    gamma: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    cm = soft_msm_cost_matrix(x, y, window, independent, c, gamma, itakura_max_slope)
    distance = cm[x.shape[-1] - 1, y.shape[-1] - 1]
    path = compute_min_return_path(cm)
    return path, distance


@njit(cache=True, fastmath=True)
def _soft_msm_cost_matrix_with_arr_univariate(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, gamma: float, c: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size, y_size))
    move_arr = np.full((x_size, y_size), np.inf)
    split_arr = np.full((x_size, y_size), np.inf)
    merge_arr = np.full((x_size, y_size), np.inf)

    for ch in range(x.shape[0]):

        _soft_msm_univariate_cost_matrix_with_arr(
            x[ch],
            y[ch],
            bounding_matrix,
            gamma,
            c,
            cost_matrix,
            move_arr,
            split_arr,
            merge_arr,
        )

    return cost_matrix, split_arr, merge_arr, move_arr


@njit(cache=True, fastmath=True)
def _soft_msm_univariate_cost_matrix_with_arr(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    c: float,
    cost_matrix,
    move_arr,
    split_arr,
    merge_arr,
):
    """Compute soft msm cost matrix and arrays for univariate time series.

    This method MUTATES: cost_matrix, move_arr, split_arr, merge_arr.

    This isn't intended for public consumption so I decided it's probably ok to
    mutate these arrays in-place. This is a performance optimisation to avoid
    unnecessary memory allocations.
    """
    x_size = x.shape[0]
    y_size = y.shape[0]

    cost_matrix[0, 0] += abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            split_cost = cost_matrix[i - 1, 0] + _cost_independent(
                x[i], x[i - 1], y[0], c
            )
            split_arr[i, 0] += split_cost
            cost_matrix[i, 0] += _softmin3(split_cost, np.inf, np.inf, gamma)

    for j in range(1, y_size):
        if bounding_matrix[0, j]:
            merge_cost = cost_matrix[0, j - 1] + _cost_independent(
                y[j], x[0], y[j - 1], c
            )
            merge_arr[0, j] += merge_cost
            cost_matrix[0, j] += _softmin3(merge_cost, np.inf, np.inf, gamma)
    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                mv = cost_matrix[i - 1, j - 1] + abs(x[i] - y[j])
                move_arr[i, j] += mv

                sp = cost_matrix[i - 1, j] + _cost_independent(x[i], x[i - 1], y[j], c)
                split_arr[i, j] += sp

                mg = cost_matrix[i, j - 1] + _cost_independent(y[j], x[i], y[j - 1], c)
                merge_arr[i, j] += mg

                cost_matrix[i, j] += _softmin3(mv, sp, mg, gamma)


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
        _soft_msm_cost_matrix_with_arr_univariate,
        gamma=gamma,
        window=window,
        itakura_max_slope=itakura_max_slope,
        c=c,
    )
