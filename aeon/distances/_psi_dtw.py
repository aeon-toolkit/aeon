__author__ = ["chrisholder", "TonyBagnall"]

import math
from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance
from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def psi_dtw_distance(x: np.ndarray, y: np.ndarray, window: float = None,
                     r: float = 0.2) -> float:
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same size")
    if r < 0 or r > 1:
        raise ValueError("r must be between 0 and 1")
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _psi_dtw_distance(_x, _y, bounding_matrix, r)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _psi_dtw_distance(x, y, bounding_matrix, r)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def psi_dtw_cost_matrix(x: np.ndarray, y: np.ndarray, window: float = None,
                        r: float = 0.2) -> np.ndarray:
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same size")
    if r < 0 or r > 1:
        raise ValueError("r must be between 0 and 1")
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _psi_dtw_cost_matrix(_x, _y, bounding_matrix, r)[1:, 1:]
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _psi_dtw_cost_matrix(x, y, bounding_matrix, r)[1:, 1:]
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _psi_dtw_distance(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, r: float
) -> float:
    cost_matrix = _psi_dtw_cost_matrix(x, y, bounding_matrix, r)
    r = math.ceil(x.shape[1] * r)
    return min(np.min(cost_matrix[-1, -r - 1:]), np.min(cost_matrix[-r - 1:, -1]))


@njit(cache=True, fastmath=True)
def _psi_dtw_cost_matrix(
        x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, r: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    r = math.ceil(x_size * r)
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0:r] = 0
    cost_matrix[0:r, 0] = 0

    for i in range(1, x_size + 1):
        beg_w = max(1, i - r)
        end_w = min(i + r + 1, y_size + 1)
        for j in range(beg_w, end_w):
            if bounding_matrix[i - 1, j - 1]:
                cost_matrix[i, j] = _univariate_squared_distance(
                    x[:, i - 1], y[:, j - 1]
                ) + min(
                    cost_matrix[i - 1, j],
                    cost_matrix[i, j - 1],
                    cost_matrix[i - 1, j - 1],
                )
    return cost_matrix


@njit(cache=True, fastmath=True)
def psi_dtw_pairwise_distance(
        X: np.ndarray, y: np.ndarray = None, window: float = None, r: float = 0.2
) -> np.ndarray:
    if r < 0 or r > 1:
        raise ValueError("r must be between 0 and 1")
    if y is None:
        # To self
        if X.ndim == 3:
            return _psi_dtw_pairwise_distance(X, window, r)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _psi_dtw_pairwise_distance(_X, window, r)
        raise ValueError("x and y must be 2D or 3D arrays")
    if X.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same size")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _psi_dtw_from_multiple_to_multiple_distance(_x, _y, window, r)


@njit(cache=True, fastmath=True)
def _psi_dtw_pairwise_distance(X: np.ndarray, window: float, r: float) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _psi_dtw_distance(X[i], X[j], bounding_matrix, r)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _psi_dtw_from_multiple_to_multiple_distance(
        x: np.ndarray, y: np.ndarray, window: float, r: float = 0.2
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _psi_dtw_distance(x[i], y[j], bounding_matrix, r)
    return distances


@njit(cache=True, fastmath=True)
def psi_dtw_alignment_path(
        x: np.ndarray, y: np.ndarray, window: float = None, r: float = 0.2
) -> Tuple[List[Tuple[int, int]], float]:
    if r < 0 or r > 1:
        raise ValueError("r must be between 0 and 1")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same size")
    cost_matrix = psi_dtw_cost_matrix(x, y, window, r)[1:, 1:]
    r = math.ceil(x.shape[-1] * r)
    return (
        compute_min_return_path(cost_matrix),
        min(np.min(cost_matrix[-1, -r - 1:]), np.min(cost_matrix[-r - 1:, -1]))
    )
