__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit, prange, set_num_threads
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_distance_utils import (
    _compute_soft_gradient,
    _soft_min,
    _soft_min_with_arrs,
    _univariate_euclidean_distance_with_difference,
)
from aeon.distances.pointwise._euclidean import _univariate_euclidean_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation import check_n_jobs
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    gamma: float = 1.0,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_erp_distance(_x, _y, bounding_matrix, g, g_arr, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_erp_distance(x, y, bounding_matrix, g, g_arr, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_erp_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    gamma: float = 1.0,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_erp_cost_matrix(_x, _y, bounding_matrix, g, g_arr, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_erp_cost_matrix(x, y, bounding_matrix, g, g_arr, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    g: float,
    g_arr: Optional[np.ndarray],
    gamma: float,
) -> float:
    return abs(
        _soft_erp_cost_matrix(x, y, bounding_matrix, g, g_arr, gamma)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    )


@njit(cache=True, fastmath=True)
def _soft_erp_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    g: float,
    g_arr: Optional[np.ndarray],
    gamma: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    gx_distance, x_sum = _precompute_g(x, g, g_arr)
    gy_distance, y_sum = _precompute_g(y, g, g_arr)

    cost_matrix[1:, 0] = x_sum
    cost_matrix[0, 1:] = y_sum
    cost_matrix[0, 0] = 0.0

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                cost_matrix[i, j] = _soft_min(
                    cost_matrix[i - 1, j - 1]
                    + _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]),
                    cost_matrix[i - 1, j] + gx_distance[i - 1],
                    cost_matrix[i, j - 1] + gy_distance[j - 1],
                    gamma,
                )

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _precompute_g(
    x: np.ndarray, g: float, g_array: Optional[np.ndarray]
) -> tuple[np.ndarray, float]:
    gx_distance = np.zeros(x.shape[1])
    if g_array is None:
        g_arr = np.full(x.shape[0], g)
    else:
        if g_array.shape[0] != x.shape[0]:
            raise ValueError("g must be a float or an array with shape (x.shape[0],)")
        g_arr = g_array
    x_sum = 0

    for i in range(x.shape[1]):
        temp = _univariate_euclidean_distance(x[:, i], g_arr)
        gx_distance[i] = temp
        x_sum += temp
    return gx_distance, x_sum


def soft_erp_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    gamma: float = 1.0,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    n_jobs = check_n_jobs(n_jobs)
    set_num_threads(n_jobs)
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )
    if y is None:
        return _soft_erp_pairwise_distance(
            _X, window, g, g_arr, itakura_max_slope, unequal_length, gamma
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_erp_from_multiple_to_multiple_distance(
        _X, _y, window, g, g_arr, itakura_max_slope, unequal_length, gamma
    )


@njit(cache=True, fastmath=True, parallel=True)
def _soft_erp_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    g: float,
    g_arr: Optional[np.ndarray],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
    gamma: float,
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
            distances[i, j] = _soft_erp_distance(
                x1, x2, bounding_matrix, g, g_arr, gamma
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _soft_erp_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    g: float,
    g_arr: Optional[np.ndarray],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
    gamma: float,
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
            distances[i, j] = _soft_erp_distance(
                x1, y1, bounding_matrix, g, g_arr, gamma
            )
    return distances


@njit(cache=True, fastmath=True)
def soft_erp_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    gamma: float = 1.0,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    cost_matrix = soft_erp_cost_matrix(
        x,
        y,
        window=window,
        g=g,
        g_arr=g_arr,
        itakura_max_slope=itakura_max_slope,
        gamma=gamma,
    )
    return (
        compute_min_return_path(cost_matrix),
        abs(cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1]),
    )


@njit(cache=True, fastmath=True)
def _soft_erp_cost_matrix_with_arrs(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)

    gx_distance, x_sum = _precompute_g(x, g, g_arr)
    gy_distance, y_sum = _precompute_g(y, g, g_arr)

    cost_matrix[1:, 0] = x_sum
    cost_matrix[0, 1:] = y_sum
    cost_matrix[0, 0] = 0.0

    diagonal_arr = np.full((x_size, y_size), np.inf)
    vertical_arr = np.full((x_size, y_size), np.inf)
    horizontal_arr = np.full((x_size, y_size), np.inf)

    diff_dist_matrix = np.zeros((x_size, y_size))

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                current_dist, difference = (
                    _univariate_euclidean_distance_with_difference(x[:, i], y[:, j])
                )
                diff_dist_matrix[i, j] = difference

                cost_matrix[i + 1, j + 1] = _soft_min_with_arrs(
                    cost_matrix[i, j] + current_dist,
                    cost_matrix[i, j + 1] + gx_distance[i],
                    cost_matrix[i + 1, j] + gy_distance[j],
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


def soft_erp_gradient(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    gamma: float = 1.0,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    return _compute_soft_gradient(
        x,
        y,
        _soft_erp_cost_matrix_with_arrs,
        gamma=gamma,
        window=window,
        itakura_max_slope=itakura_max_slope,
        g=g,
        g_arr=g_arr,
    )[0:2]
