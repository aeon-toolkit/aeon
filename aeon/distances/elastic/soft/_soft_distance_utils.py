__maintainer__ = []
from typing import Optional

import numpy as np
from numba import njit

from aeon.distances.elastic._bounding_matrix import create_bounding_matrix


@njit(fastmath=True, cache=True)
def _soft_min(a, b, c, gamma):
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
    exp_sum = np.exp(a - max_val) + np.exp(b - max_val) + np.exp(c - max_val)
    return -gamma * (np.log(exp_sum) + max_val)


@njit(cache=True, fastmath=True)
def _soft_min_with_arrs(
    diagonal_value: float,
    vertical_value: float,
    horizontal_value: float,
    gamma: float,
    diagonal_arr: np.ndarray,
    vertical_arr: np.ndarray,
    horizontal_arr: np.ndarray,
    i: int,
    j: int,
) -> float:
    """Compute softmin and store directional contributions in arrays.

    Parameters
    ----------
    diagonal_value, vertical_value, horizontal_value : float
        Input values representing diagonal (move), vertical (split),
        and horizontal (merge) operations respectively
    gamma : float
        Softmin smoothing parameter
    diagonal_arr : np.ndarray
        Array to store move operation contributions
    vertical_arr : np.ndarray
        Array to store split operation contributions
    horizontal_arr : np.ndarray
        Array to store merge operation contributions
    i, j : int
        Current position in the arrays to update

    Returns
    -------
    float
        Softmin value of a, b, c
    """
    a_scaled = diagonal_value / -gamma
    b_scaled = vertical_value / -gamma
    c_scaled = horizontal_value / -gamma

    max_val = max(a_scaled, b_scaled, c_scaled)

    exp_a = np.exp(a_scaled - max_val)
    exp_b = np.exp(b_scaled - max_val)
    exp_c = np.exp(c_scaled - max_val)
    exp_sum = exp_a + exp_b + exp_c

    diagonal_arr[i, j] = exp_a / exp_sum
    vertical_arr[i, j] = exp_b / exp_sum
    horizontal_arr[i, j] = exp_c / exp_sum

    return -gamma * (np.log(exp_sum) + max_val)


@njit(cache=True, fastmath=True)
def _jacobian_product_squared_euclidean(
    X: np.ndarray, Y: np.ndarray, E: np.ndarray, diff_matrix: np.ndarray
):
    m = X.shape[1]
    n = Y.shape[1]
    d = X.shape[0]

    product = np.zeros((d, m))

    for i in range(m):
        for j in range(n):
            for k in range(d):
                product[k, i] += E[i, j] * 2 * (diff_matrix[i, j])
                # product[k, i] += E[i, j] * 2 * (X[k, i] - Y[k, j])
    return product


@njit(cache=True, fastmath=True)
def _jacobian_product_absolute_distance(
    X: np.ndarray, Y: np.ndarray, E: np.ndarray, diff_matrix: np.ndarray
) -> np.ndarray:
    d, m = X.shape[0], X.shape[1]
    n = Y.shape[1]

    product = np.zeros((d, m), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            e_ij = E[i, j]
            if e_ij != 0.0:
                for k in range(d):
                    product[k, i] += e_ij * np.sign(diff_matrix[i, j])
                    # product[k, i] += e_ij * np.sign(X[k, i] - Y[k, j])

    return product


FLOAT_EPS = np.finfo(np.float64).eps


@njit(cache=True, fastmath=True)
def _jacobian_product_euclidean(
    X: np.ndarray, Y: np.ndarray, E: np.ndarray, diff_matrix: np.ndarray
) -> np.ndarray:
    d, m = X.shape
    n = Y.shape[1]

    product = np.zeros((d, m), dtype=np.float64)
    diff_vector = np.empty(d, dtype=np.float64)

    for i in range(m):
        for j in range(n):
            e_ij = E[i, j]
            if e_ij == 0.0:
                continue

            dist_squared = 0.0
            for k in range(d):
                diff_vector[k] = diff_matrix[i, j] * diff_matrix[i, j]
                dist_squared += diff_vector[k] * diff_vector[k]

            if dist_squared > FLOAT_EPS:
                dist = np.sqrt(dist_squared)
                factor = e_ij / dist
                for k in range(d):
                    product[k, i] += factor * diff_vector[k]

    return product


# @njit(cache=True, fastmath=True)
# def _jacobian_product_euclidean(
#     X: np.ndarray, Y: np.ndarray, E: np.ndarray, diff_matrix: np.ndarray
# ) -> np.ndarray:
#     d, m = X.shape[0], X.shape[1]
#     n = Y.shape[1]
#
#     product = np.zeros((d, m), dtype=np.float64)
#
#     for i in range(m):
#         for j in range(n):
#             e_ij = E[i, j]
#             if e_ij > FLOAT_EPS:
#                 dist_squared = diff_matrix[i, j] * diff_matrix[i, j]
#                 if dist_squared > FLOAT_EPS:
#                     inv_dist = 1.0 / np.sqrt(dist_squared)
#                     for k in range(d):
#                         product[k, i] += e_ij * diff_matrix[i, j] * inv_dist
#
#     return product


def _compute_soft_gradient(
    x: np.ndarray,
    y: np.ndarray,
    cost_matrix_with_arrs_func: callable,
    gamma: float = 1.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    **kwargs,
) -> tuple[np.ndarray, float]:
    if gamma <= 0:
        raise ValueError("gamma must be greater than 0 for this method")
    if x.ndim == 1 or y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
    else:
        _x = x
        _y = y
    bounding_matrix = create_bounding_matrix(
        _x.shape[1], _y.shape[1], window, itakura_max_slope
    )

    return _compute_soft_gradient_with_diff_dist_matrix(
        _x, _y, cost_matrix_with_arrs_func, gamma, bounding_matrix, **kwargs
    )[0:2]


def _compute_soft_gradient_with_diff_dist_matrix(
    x: np.ndarray,
    y: np.ndarray,
    cost_matrix_with_arrs_func: callable,
    gamma: float,
    bounding_matrix: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, float, np.ndarray]:
    cost_matrix, diagonal_arr, vertical_arr, horizontal_arr, diff_dist_matrix = (
        cost_matrix_with_arrs_func(
            x, y, bounding_matrix=bounding_matrix, gamma=gamma, **kwargs
        )
    )
    return (
        _soft_gradient_with_arrs(
            cost_matrix, diagonal_arr, vertical_arr, horizontal_arr
        ),
        cost_matrix[-1, -1],
        diff_dist_matrix,
    )


@njit(cache=True, fastmath=True)
def _soft_gradient_with_arrs(
    cost_matrix: np.ndarray,
    diagonal_move_arr: np.ndarray,
    vertical_move_arr: np.ndarray,
    horizontal_move_arr: np.ndarray,
) -> np.ndarray:
    m, n = cost_matrix.shape
    E = np.zeros((m, n), dtype=float)

    E[m - 1, n - 1] = 1.0

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            E_ij = E[i, j]

            if i + 1 < m:
                E_ij += E[i + 1, j] * vertical_move_arr[i + 1, j]

            if j + 1 < n:
                E_ij += E[i, j + 1] * horizontal_move_arr[i, j + 1]

            if (i + 1 < m) and (j + 1 < n):
                E_ij += E[i + 1, j + 1] * diagonal_move_arr[i + 1, j + 1]

            E[i, j] = E_ij

    return E


@njit(cache=True, fastmath=True)
def _univariate_squared_distance_with_difference(x: np.ndarray, y: np.ndarray):
    """Return the squared distance and difference.

    The intention is to use this to avoid recomputing x[i] - y[i] since you need to
    do this in the Jacobian as well.
    """
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    difference_sum = 0.0
    for i in range(min_length):
        difference = x[i] - y[i]
        difference_sum += difference
        distance += difference * difference
    return distance, difference


@njit(cache=True, fastmath=True)
def _univariate_euclidean_distance_with_difference(x: np.ndarray, y: np.ndarray):
    distance, difference = _univariate_squared_distance_with_difference(x, y)
    return np.sqrt(distance), difference


# This was the DTW specific adaptation can uncomment later to compare speed of running
# this again my implementation
# @njit(cache=True, fastmath=True)
# def _soft_gradient(
#     distance_matrix: np.ndarray, cost_matrix: np.ndarray, gamma: float
# ) -> np.ndarray:
#     m, n = distance_matrix.shape
#     E = np.zeros((m, n), dtype=float)
#
#     E[m - 1, n - 1] = 1.0
#
#     for i in range(m - 1, -1, -1):
#         for j in range(n - 1, -1, -1):
#             r_ij = cost_matrix[i, j]
#             E_ij = E[i, j]
#
#             if i + 1 < m:
#                 w_horizontal = np.exp(
#                     (cost_matrix[i + 1, j] - r_ij - distance_matrix[i + 1, j]) / gamma
#                 )
#                 E_ij += E[i + 1, j] * w_horizontal
#
#             if j + 1 < n:
#                 w_vertical = np.exp(
#                     (cost_matrix[i, j + 1] - r_ij - distance_matrix[i, j + 1]) / gamma
#                 )
#                 E_ij += E[i, j + 1] * w_vertical
#
#             if (i + 1 < m) and (j + 1 < n):
#                 w_diag = np.exp(
#                     (cost_matrix[i + 1, j + 1] - r_ij - distance_matrix[i + 1, j + 1])
#                     / gamma
#                 )
#                 E_ij += E[i + 1, j + 1] * w_diag
#
#             E[i, j] = E_ij
#
#     return E
#
# This part needs to go into the DTW file
# @njit(cache=True, fastmath=True)
# def _soft_dtw_cost_matrix_return_dist_matrix(
#         x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, gamma: float
# ) -> tuple[np.ndarray, np.ndarray]:
#     x_size = x.shape[1]
#     y_size = y.shape[1]
#     cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
#     cost_matrix[0, 0] = 0.0
#     dist_matrix = np.zeros((x_size, y_size))
#
#     for i in range(1, x_size + 1):
#         for j in range(1, y_size + 1):
#             if bounding_matrix[i - 1, j - 1]:
#                 dist = _univariate_squared_distance(x[:, i - 1], y[:, j - 1])
#                 dist_matrix[i - 1, j - 1] = dist
#                 cost_matrix[i, j] = dist + _softmin3(
#                     cost_matrix[i - 1, j],
#                     cost_matrix[i - 1, j - 1],
#                     cost_matrix[i, j - 1],
#                     gamma,
#                 )
#     return cost_matrix[1:, 1:], dist_matrix
