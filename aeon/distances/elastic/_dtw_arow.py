r"""Dynamic time warping (DTW) between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.pointwise._squared import _univariate_squared_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True)
def dtw_arow_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        return _dtw_arow_distance(_x, _y)
    if x.ndim == 2 and y.ndim == 2:
        return _dtw_arow_distance(x, y)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True)
def dtw_arow_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        return np.sqrt(_dtw_arow_cost_matrix(_x, _y))
    if x.ndim == 2 and y.ndim == 2:
        return np.sqrt(_dtw_arow_cost_matrix(x, y))
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True)
def _dtw_arow_distance(x: np.ndarray, y: np.ndarray) -> float:
    M1av = _total_non_nan(x)
    M2av = _total_non_nan(y)
    gamma = (x.shape[1] + y.shape[1]) / (M1av + M2av)
    return np.sqrt(gamma * _dtw_arow_cost_matrix(x, y)[x.shape[1], y.shape[1]])


@njit(cache=True)
def _check_nan(x: np.ndarray) -> bool:
    return np.sum(np.isnan(x)) > 0


@njit(cache=True, fastmath=True)
def _cost_function(x: np.ndarray, y: np.ndarray) -> float:
    if _check_nan(x) or _check_nan(y):
        return 0
    return _univariate_squared_distance(x, y)


@njit(cache=True)
def _total_non_nan(x: np.ndarray) -> float:
    if x.shape[0] == 1:
        return np.sum(~np.isnan(x[0]))
    else:
        return np.sum(np.sum(np.isnan(x), axis=0) == 0)


@njit(cache=True)
def _dtw_arow_cost_path_helper(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[1]
    m = y.shape[1]
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    phi = np.full((n + 1, m + 1), -1)

    cost_matrix[0, 0] = 0

    x_avail = np.zeros(n, dtype=np.bool_)
    for k in range(n):
        x_avail[k] = not _check_nan(x[:, k])

    y_avail = np.zeros(m, dtype=np.bool_)
    for k in range(m):
        y_avail[k] = not _check_nan(y[:, k])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ev = np.inf
            if i > 1:
                if y_avail[j - 1] and x_avail[i - 1] and x_avail[i - 2]:
                    ev = 0

            eh = np.inf
            if j > 1:
                if x_avail[i - 1] and y_avail[j - 1] and y_avail[j - 2]:
                    eh = 0

            cost_diag = cost_matrix[i - 1, j - 1]
            cost_vert = cost_matrix[i - 1, j] + ev
            cost_horiz = cost_matrix[i, j - 1] + eh

            options = np.array([cost_diag, cost_vert, cost_horiz])
            best_prev = np.min(options)

            if np.isinf(best_prev):
                cost_matrix[i, j] = np.inf
            else:
                current_cost = _cost_function(x[:, i - 1], y[:, j - 1])
                cost_matrix[i, j] = current_cost + best_prev
                phi[i, j] = np.argmin(options)

    return cost_matrix, phi


@njit(cache=True)
def _dtw_arow_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _dtw_arow_cost_path_helper(x, y)[0]


@njit(cache=True)
def dtw_arow_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, _ = _convert_collection_to_numba_list(X, "X", multivariate_conversion)

    if y is None:
        # To self
        return _dtw_arow_pairwise_distance(_X)
    _y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
    return _dtw_arow_from_multiple_to_multiple_distance(_X, _y)


@njit(cache=True)
def _dtw_arow_pairwise_distance(X: NumbaList[np.ndarray]) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            distances[i, j] = _dtw_arow_distance(x1, x2)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True)
def _dtw_arow_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray], y: NumbaList[np.ndarray]
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            distances[i, j] = _dtw_arow_distance(x1, y1)
    return distances


@njit(cache=True)
def dtw_arow_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[list[tuple[int, int]], float]:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        return _dtw_arow_alignment_path(_x, _y)
    if x.ndim == 2 and y.ndim == 2:
        return _dtw_arow_alignment_path(x, y)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True)
def _dtw_arow_alignment_path(
    x: np.ndarray, y: np.ndarray
) -> tuple[list[tuple[int, int]], float]:
    i, j = x.shape[1], y.shape[1]
    path = []
    phi = _dtw_arow_cost_path_helper(x, y)[1]
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = phi[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i = i - 1
        elif step == 2:
            j = j - 1
        else:
            break
    path.reverse()

    return path, dtw_arow_distance(x, y)
