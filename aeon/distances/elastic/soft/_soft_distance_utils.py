__maintainer__ = []

import math
from typing import Optional

import numpy as np
from numba import njit

from aeon.distances.elastic._bounding_matrix import create_bounding_matrix


@njit(fastmath=True, cache=True)
def _softmin3(a, b, c, gamma):
    r"""Compute softmin of 3 input variables with parameter gamma.

    This code is adapted from tslearn.

    Parameters
    ----------
    a : float
        First input variable.
    b : float
        Second input variable.
    c : float
        Third input variable.
    gamma : float
        Softmin parameter.

    Returns
    -------
    float
        Softmin of a, b, c.
    """
    a /= -gamma
    b /= -gamma
    c /= -gamma
    max_val = max(a, b, c)
    tmp = np.exp(a - max_val) + np.exp(b - max_val) + np.exp(c - max_val)
    return -gamma * (np.log(tmp) + max_val)


@njit(cache=True, fastmath=True)
def _soft_gradient(
    distance_matrix: np.ndarray, cost_matrix: np.ndarray, gamma: float
) -> np.ndarray:
    m, n = distance_matrix.shape
    E = np.zeros((m, n), dtype=float)

    inv_gamma = 1.0 / gamma

    E[m - 1, n - 1] = 1.0

    for i in range(m - 1, -1, -1):
        rowE = E[i]
        rowC = cost_matrix[i]
        for j in range(n - 1, -1, -1):
            r_ij = rowC[j]
            E_ij = rowE[j]

            if i + 1 < m:
                r_down = cost_matrix[i + 1, j]
                d_down = distance_matrix[i + 1, j]
                w_down = np.exp((r_down - r_ij - d_down) * inv_gamma)
                E_ij += E[i + 1, j] * w_down

            if j + 1 < n:
                r_right = cost_matrix[i, j + 1]
                d_right = distance_matrix[i, j + 1]
                w_right = np.exp((r_right - r_ij - d_right) * inv_gamma)
                E_ij += E[i, j + 1] * w_right

            if (i + 1 < m) and (j + 1 < n):
                r_diag = cost_matrix[i + 1, j + 1]
                d_diag = distance_matrix[i + 1, j + 1]
                w_diag = np.exp((r_diag - r_ij - d_diag) * inv_gamma)
                E_ij += E[i + 1, j + 1] * w_diag

            rowE[j] = E_ij

    return E


@njit(cache=True, fastmath=True)
def _jacobian_product_squared_euclidean(X, Y, E):
    m = X.shape[1]
    n = Y.shape[1]
    d = X.shape[0]

    product = np.zeros((d, m))

    for i in range(m):
        for j in range(n):
            for k in range(d):
                product[k, i] += E[i, j] * 2 * (X[k, i] - Y[k, j])
    return product


@njit(cache=True, fastmath=True)
def _jacobian_product_absolute_distance(
    X: np.ndarray, Y: np.ndarray, E: np.ndarray
) -> np.ndarray:
    d, m = X.shape[0], X.shape[1]
    n = Y.shape[1]

    product = np.zeros((d, m), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            e_ij = E[i, j]
            if e_ij != 0.0:
                for k in range(d):
                    diff = X[k, i] - Y[k, j]
                    if diff > 0:
                        sign_val = 1.0
                    elif diff < 0:
                        sign_val = -1.0
                    else:
                        sign_val = 0.0
                    product[k, i] += e_ij * sign_val

    return product


def _compute_soft_gradient(
    x: np.ndarray,
    y: np.ndarray,
    cost_matrix_with_arrs_func: callable,
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    **kwargs,
) -> tuple[np.ndarray, float]:
    if x.ndim == 1 or y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
    else:
        _x = x
        _y = y
    bounding_matrix = create_bounding_matrix(
        _x.shape[1], _y.shape[1], window, itakura_max_slope
    )

    cost_matrix, vertical_arr, horizontal_arr, diagonal_arr = (
        cost_matrix_with_arrs_func(
            _x, _y, bounding_matrix=bounding_matrix, gamma=gamma, **kwargs
        )
    )
    return (
        _soft_gradient_with_arrs(
            cost_matrix, vertical_arr, horizontal_arr, diagonal_arr, gamma
        ),
        cost_matrix[-1, -1],
    )


@njit(cache=True, fastmath=True)
def _soft_gradient_with_arrs(
    cost_matrix: np.ndarray,
    vertical_move_arr: np.ndarray,
    horizontal_move_arr: np.ndarray,
    diagonal_move_arr: np.ndarray,
    gamma: float,
) -> np.ndarray:
    m, n = cost_matrix.shape
    E = np.zeros((m, n), dtype=np.float64)
    E[m - 1, n - 1] = 1.0
    inv_gamma = 1.0 / gamma

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            e_ij = E[i, j]
            if e_ij == 0.0:
                continue

            c_ij = cost_matrix[i, j]

            dx = horizontal_move_arr[i, j]
            dy = vertical_move_arr[i, j]
            mt = diagonal_move_arr[i, j]

            w_dx = 0.0
            w_dy = 0.0
            w_mt = 0.0
            if not np.isinf(dx):
                w_dx = math.exp((dx - c_ij) * -inv_gamma)
            if not np.isinf(dy):
                w_dy = math.exp((dy - c_ij) * -inv_gamma)
            if not np.isinf(mt):
                w_mt = math.exp((mt - c_ij) * -inv_gamma)

            denom = w_dx + w_dy + w_mt
            if denom < 1e-15:
                continue

            alpha_dx = w_dx / denom
            alpha_dy = w_dy / denom
            alpha_mt = w_mt / denom

            if i > 0:
                E[i - 1, j] += e_ij * alpha_dx
            if j > 0:
                E[i, j - 1] += e_ij * alpha_dy
            if i > 0 and j > 0:
                E[i - 1, j - 1] += e_ij * alpha_mt

    return E
