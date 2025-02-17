r"""Soft dynamic time warping (soft-ADTW) between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_distance_utils import (
    _compute_soft_gradient,
    _soft_min,
    _soft_min_with_arrs,
    _univariate_squared_distance_with_difference,
)
from aeon.distances.pointwise._squared import _univariate_squared_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate

MAX_FLOAT = np.finfo(np.float64).max


@njit(cache=True, fastmath=True)
def soft_adtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_adtw_distance(_x, _y, bounding_matrix, gamma, warp_penalty)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_adtw_distance(x, y, bounding_matrix, gamma, warp_penalty)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_adtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_adtw_cost_matrix(_x, _y, bounding_matrix, gamma, warp_penalty)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_adtw_cost_matrix(x, y, bounding_matrix, gamma, warp_penalty)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_adtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    warp_penalty: float,
) -> float:
    return abs(
        _soft_adtw_cost_matrix(x, y, bounding_matrix, gamma, warp_penalty)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    )


@njit(cache=True, fastmath=True)
def _soft_adtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    warp_penalty: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[:, i], y[:, j]
                ) + _soft_min(
                    cost_matrix[i, j],
                    cost_matrix[i, j + 1] + warp_penalty,
                    cost_matrix[i + 1, j] + warp_penalty,
                    gamma,
                )
    return cost_matrix[1:, 1:]


def soft_adtw_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _soft_adtw_pairwise_distance(
            _X, window, itakura_max_slope, unequal_length, gamma, warp_penalty
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_adtw_from_multiple_to_multiple_distance(
        _X, _y, window, itakura_max_slope, unequal_length, gamma, warp_penalty
    )


@njit(cache=True, fastmath=True)
def _soft_adtw_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
    gamma: float,
    warp_penalty: float = 1.0,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_adtw_distance(
                x1, x2, bounding_matrix, gamma, warp_penalty
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _soft_adtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
    gamma: float,
    warp_penalty: float,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_adtw_distance(
                x1, y1, bounding_matrix, gamma, warp_penalty
            )
    return distances


@njit(cache=True, fastmath=True)
def soft_adtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> tuple[list[tuple[int, int]], float]:
    cost_matrix = soft_adtw_cost_matrix(
        x, y, gamma, window, itakura_max_slope, warp_penalty
    )
    return (
        compute_min_return_path(cost_matrix),
        abs(cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1]),
    )


@njit(cache=True, fastmath=True)
def _soft_adtw_cost_matrix_with_arrs(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    warp_penalty: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    diagonal_arr = np.full((x_size, y_size), np.inf)
    vertical_arr = np.full((x_size, y_size), np.inf)
    horizontal_arr = np.full((x_size, y_size), np.inf)

    diff_dist_matrix = np.zeros((x_size, y_size))

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                current_dist, difference = _univariate_squared_distance_with_difference(
                    x[:, i], y[:, j]
                )
                diff_dist_matrix[i, j] = difference
                cost_matrix[i + 1, j + 1] = current_dist + _soft_min_with_arrs(
                    cost_matrix[i, j],
                    cost_matrix[i, j + 1] + warp_penalty,
                    cost_matrix[i + 1, j] + warp_penalty,
                    gamma,
                    diagonal_arr,
                    vertical_arr,
                    horizontal_arr,
                    i,
                    j,
                )

    return (
        cost_matrix[1:, 1:],
        diagonal_arr,
        vertical_arr,
        horizontal_arr,
        diff_dist_matrix,
    )


def soft_adtw_gradient(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> tuple[np.ndarray, float]:
    return _compute_soft_gradient(
        x,
        y,
        _soft_adtw_cost_matrix_with_arrs,
        gamma=gamma,
        window=window,
        itakura_max_slope=itakura_max_slope,
        warp_penalty=warp_penalty,
    )[0:2]
