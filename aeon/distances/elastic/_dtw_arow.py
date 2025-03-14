r"""Dynamic time warping (DTW) between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np

# from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.pointwise._squared import _univariate_squared_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


def dtw_arow_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        return _dtw_arow_distance(_x, _y)
    if x.ndim == 2 and y.ndim == 2:
        return _dtw_arow_distance(x, y)
    raise ValueError("x and y must be 1D or 2D")


def dtw_arow_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        return _dtw_arow_cost_matrix(_x, _y)
    if x.ndim == 2 and y.ndim == 2:
        return _dtw_arow_cost_matrix(x, y)
    raise ValueError("x and y must be 1D or 2D")


def _dtw_arow_distance(x: np.ndarray, y: np.ndarray) -> float:
    M1av = _total_non_nan(x)
    M2av = _total_non_nan(y)
    gamma = (x.shape[1] + y.shape[1]) / (M1av + M2av) if (M1av + M2av) > 0 else 1
    return np.sqrt(gamma * _dtw_arow_cost_matrix(x, y)[x.shape[1], y.shape[1]])


def _check_nan(x: np.ndarray) -> bool:
    return np.isnan(x).any()


def _cost_function(x: np.ndarray, y: np.ndarray) -> float:
    if _check_nan(x) or _check_nan(y):
        return 0
    return _univariate_squared_distance(x, y)


def _total_non_nan(x: np.ndarray) -> float:
    if x.shape[0] == 1:
        return np.sum(~np.isnan(x[0]))
    else:
        return np.sum(~np.all(np.isnan(x), axis=0))


def _dtw_arow_cost_path_helper(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[1]
    m = y.shape[1]
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    phi = np.full((n + 1, m + 1), -1)

    cost_matrix[0, :] = 0
    cost_matrix[:, :] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if j > 1:
                if (
                    _check_nan(x[:, i - 1])
                    or _check_nan(y[:, j - 1])
                    or _check_nan(y[:, j - 2])
                ):
                    eh = np.inf
                else:
                    eh = 0
            else:
                if _check_nan(x[:, i - 1]) or _check_nan(y[:, j - 1]):
                    eh = np.inf
                else:
                    eh = 0

            if i > 1:
                if (
                    _check_nan(x[:, i - 1])
                    or _check_nan(x[:, i - 2])
                    or _check_nan(y[:, j - 1])
                ):
                    ev = np.inf
                else:
                    ev = 0
            else:
                if _check_nan(x[:, i - 1]) or _check_nan(y[:, j - 1]):
                    ev = np.inf
                else:
                    ev = 0

            current_cost = _cost_function(x[:, i - 1], y[:, j - 1])
            cost_diag = cost_matrix[i - 1, j - 1]
            cost_horiz = cost_matrix[i, j - 1] + eh
            cost_vert = cost_matrix[i - 1, j] + ev

            best_prev = min(cost_diag, cost_horiz, cost_vert)
            cost_matrix[i, j] = current_cost + best_prev

            if best_prev == cost_diag:
                phi[i, j] = 0
            elif best_prev == cost_horiz:
                phi[i, j] = 1
            else:
                phi[i, j] = 2

    return cost_matrix, phi


def _dtw_arow_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _dtw_arow_cost_path_helper(x, y)[0]


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


def _dtw_arow_pairwise_distance(X: NumbaList[np.ndarray]) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            distances[i, j] = _dtw_arow_distance(x1, x2)
            distances[j, i] = distances[i, j]

    return distances


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


def _dtw_arow_alignment_path(
    x: np.ndarray, y: np.ndarray
) -> tuple[list[tuple[int, int]], float]:
    i, j = x.shape[1], y.shape[1]
    path = [(x.shape[1], y.shape[1])]
    phi = _dtw_arow_cost_path_helper(x, y)[1]
    while i > 0 and j > 0:
        step = phi[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            j = j - 1
        elif step == 2:
            i = i - 1
        else:
            break
        path.append((i, j))
    path.reverse()

    return path, dtw_arow_distance(x, y)
