"""Time Warp Edit (TWE) distance between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic._twe import _pad_arrs
from aeon.distances.elastic.soft._soft_distance_utils import (
    _compute_soft_gradient,
    _soft_min,
    _soft_min_with_arrs,
    _univariate_euclidean_distance_with_difference,
)
from aeon.distances.pointwise._euclidean import _univariate_euclidean_distance
from aeon.utils._threading import threaded
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_twe_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_distance(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda, gamma
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_distance(
            _pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda, gamma
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_cost_matrix(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda, gamma
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_twe_cost_matrix(
            _pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda, gamma
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_twe_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    nu: float,
    lmbda: float,
    gamma: float,
) -> float:
    return abs(
        _soft_twe_cost_matrix(x, y, bounding_matrix, nu, lmbda, gamma)[
            x.shape[1] - 2, y.shape[1] - 2
        ]
    )


@njit(cache=True, fastmath=True)
def _soft_twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    nu: float,
    lmbda: float,
    gamma: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = 0.0

    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                current_dist = _univariate_euclidean_distance(x[:, i], y[:, j])

                del_x_dist = _univariate_euclidean_distance(x[:, i - 1], x[:, i])
                del_y_dist = _univariate_euclidean_distance(y[:, j - 1], y[:, j])

                del_x = cost_matrix[i - 1, j] + del_x_dist + del_add
                del_y = cost_matrix[i, j - 1] + del_y_dist + del_add

                match_prev = _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1])
                time_penalty = nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                match = (
                    cost_matrix[i - 1, j - 1] + current_dist + match_prev + time_penalty
                )

                cost_matrix[i, j] = _soft_min(
                    match,
                    del_x,
                    del_y,
                    gamma,
                )

    return cost_matrix[1:, 1:]


@threaded
def soft_twe_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    gamma: float = 1.0,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )
    if y is None:
        # To self
        return _soft_twe_pairwise_distance(
            _X, window, nu, lmbda, itakura_max_slope, unequal_length, gamma
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_twe_from_multiple_to_multiple_distance(
        _X, _y, window, nu, lmbda, itakura_max_slope, unequal_length, gamma
    )


@njit(cache=True, fastmath=True, parallel=True)
def _soft_twe_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
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

    # Pad the arrays before so that we don't have to redo every iteration
    padded_X = NumbaList()
    for i in range(n_cases):
        padded_X.append(_pad_arrs(X[i]))

    for i in prange(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = padded_X[i], padded_X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_twe_distance(
                x1, x2, bounding_matrix, nu, lmbda, gamma
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _soft_twe_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
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

    # Pad the arrays before so that we dont have to redo every iteration
    padded_x = NumbaList()
    for i in range(n_cases):
        padded_x.append(_pad_arrs(x[i]))

    padded_y = NumbaList()
    for i in range(m_cases):
        padded_y.append(_pad_arrs(y[i]))

    for i in prange(n_cases):
        for j in range(m_cases):
            x1, y1 = padded_x[i], padded_y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _soft_twe_distance(
                x1, y1, bounding_matrix, nu, lmbda, gamma
            )
    return distances


@njit(cache=True, fastmath=True)
def soft_twe_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    gamma: float = 1.0,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    cost_matrix = soft_twe_cost_matrix(
        x,
        y,
        window=window,
        nu=nu,
        lmbda=lmbda,
        itakura_max_slope=itakura_max_slope,
        gamma=gamma,
    )
    return (
        compute_min_return_path(cost_matrix),
        abs(cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1]),
    )


@njit(cache=True, fastmath=True)
def _soft_twe_cost_matrix_with_arrs(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    nu: float = 0.001,
    lmbda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size, y_size), np.inf)
    cost_matrix[0, 0] = 0.0

    diagonal_arr = np.zeros((x_size, y_size))
    vertical_arr = np.zeros((x_size, y_size))
    horizontal_arr = np.zeros((x_size, y_size))
    diff_dist_matrix = np.zeros((x_size, y_size))

    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                current_dist, difference = (
                    _univariate_euclidean_distance_with_difference(x[:, i], y[:, j])
                )

                diff_dist_matrix[i, j] = difference

                del_x_dist = _univariate_euclidean_distance(x[:, i - 1], x[:, i])
                del_y_dist = _univariate_euclidean_distance(y[:, j - 1], y[:, j])

                del_x = cost_matrix[i - 1, j] + del_x_dist + del_add
                del_y = cost_matrix[i, j - 1] + del_y_dist + del_add

                match_prev = _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1])
                time_penalty = nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                match = (
                    cost_matrix[i - 1, j - 1] + current_dist + match_prev + time_penalty
                )

                cost_matrix[i, j] = _soft_min_with_arrs(
                    match,
                    del_x,
                    del_y,
                    gamma,
                    diagonal_arr,
                    vertical_arr,
                    horizontal_arr,
                    i,
                    j,
                )

    return (
        cost_matrix,
        diagonal_arr,
        vertical_arr,
        horizontal_arr,
        diff_dist_matrix,
    )


def soft_twe_gradient(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    grad, dist = _compute_soft_gradient(
        _pad_arrs(x),
        _pad_arrs(y),
        _soft_twe_cost_matrix_with_arrs,
        gamma=gamma,
        window=window,
        itakura_max_slope=itakura_max_slope,
        nu=nu,
        lmbda=lmbda,
    )
    return grad[1:, 1:], dist
