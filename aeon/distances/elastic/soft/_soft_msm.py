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


# def soft_msm_gradient(
#     x: np.ndarray,
#     y: np.ndarray,
#     gamma: float = 1.0,
#     window: Optional[float] = None,
#     c: float = 1.0,
#     itakura_max_slope: Optional[float] = None,
# ) -> tuple[np.ndarray, float]:
#     return _compute_soft_gradient(
#         x,
#         y,
#         _soft_msm_cost_matrix_with_arr_independent,
#         gamma=gamma,
#         window=window,
#         itakura_max_slope=itakura_max_slope,
#         c=c,
#     )
#
# def soft_msm_gradient(
#         x: np.ndarray,
#         y: np.ndarray,
#         gamma: float = 1.0,
#         window: Optional[float] = None,
#         c: float = 1.0,
#         itakura_max_slope: Optional[float] = None,
# ) -> np.ndarray:


@njit(cache=True, fastmath=True)
def soft_msm_gradient(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Compute the gradient of Soft-MSM distance with respect to x.

    Returns
    -------
    gradient : np.ndarray
        Gradient vector of shape (len(x),)
    distance : float
        The Soft-MSM distance
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_gradient(_x, _y, bounding_matrix, c, gamma)
    elif x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_gradient(x, y, bounding_matrix, c, gamma)
    else:
        raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_msm_gradient(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    c: float,
) -> tuple[np.ndarray, float]:
    cost_matrix, move_probs, split_probs, merge_probs = (
        _soft_msm_cost_matrix_with_probs(
            x,
            y,
            bounding_matrix=bounding_matrix,
            c=c,
            gamma=gamma,
        )
    )

    alignment_matrix = _soft_msm_alignment_matrix(
        cost_matrix, move_probs, split_probs, merge_probs
    )

    return alignment_matrix, abs(cost_matrix[-1, -1])


@njit(cache=True, fastmath=True)
def _soft_msm_cost_matrix_with_probs(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Soft-MSM cost matrix and cache move probabilities.
    """
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.full((x_size, y_size), np.inf)
    move_probs = np.zeros((x_size, y_size))
    split_probs = np.zeros((x_size, y_size))
    merge_probs = np.zeros((x_size, y_size))

    # Initialize
    cost_matrix[0, 0] = np.abs(x[0, 0] - y[0, 0])

    # First column
    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            split_cost = cost_matrix[i - 1, 0] + _cost_independent(
                x[0, i], x[0, i - 1], y[0, 0], c
            )
            cost_matrix[i, 0] = split_cost
            split_probs[i, 0] = 1.0  # Only one way to get here

    # First row
    for j in range(1, y_size):
        if bounding_matrix[0, j]:
            merge_cost = cost_matrix[0, j - 1] + _cost_independent(
                y[0, j], x[0, 0], y[0, j - 1], c
            )
            cost_matrix[0, j] = merge_cost
            merge_probs[0, j] = 1.0  # Only one way to get here

    # Main computation
    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                move_cost = cost_matrix[i - 1, j - 1] + np.abs(x[0, i] - y[0, j])
                split_cost = cost_matrix[i - 1, j] + _cost_independent(
                    x[0, i], x[0, i - 1], y[0, j], c
                )
                merge_cost = cost_matrix[i, j - 1] + _cost_independent(
                    y[0, j], x[0, i], y[0, j - 1], c
                )

                # Compute soft minimum and probabilities
                cost, probs = _soft_min_with_probs(
                    move_cost, split_cost, merge_cost, gamma
                )
                cost_matrix[i, j] = cost
                move_probs[i, j] = probs[0]
                split_probs[i, j] = probs[1]
                merge_probs[i, j] = probs[2]

    return cost_matrix, move_probs, split_probs, merge_probs


@njit(cache=True, fastmath=True)
def _soft_min_with_probs(a, b, c, gamma):
    """
    Compute soft minimum and return probabilities for each input.
    """
    if gamma == 0:
        min_val = min(a, b, c)
        probs = np.zeros(3)
        if a == min_val:
            probs[0] = 1.0
        elif b == min_val:
            probs[1] = 1.0
        else:
            probs[2] = 1.0
        return min_val, probs

    # Compute soft minimum
    a_norm = a / -gamma
    b_norm = b / -gamma
    c_norm = c / -gamma
    max_val = max(a_norm, b_norm, c_norm)

    exp_a = np.exp(a_norm - max_val)
    exp_b = np.exp(b_norm - max_val)
    exp_c = np.exp(c_norm - max_val)
    exp_sum = exp_a + exp_b + exp_c

    # Probabilities
    probs = np.array([exp_a / exp_sum, exp_b / exp_sum, exp_c / exp_sum])

    soft_min_val = -gamma * (np.log(exp_sum) + max_val)

    return soft_min_val, probs


@njit(cache=True, fastmath=True)
def _soft_msm_alignment_matrix(
    cost_matrix: np.ndarray,
    move_probs: np.ndarray,
    split_probs: np.ndarray,
    merge_probs: np.ndarray,
) -> np.ndarray:
    """
    Compute alignment matrix via backpropagation.
    """
    n, m = cost_matrix.shape
    alignment = np.zeros((n, m))
    alignment[-1, -1] = 1.0

    # Backpropagate
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if i < n - 1 and j < m - 1:
                # Contribution from diagonal move to (i+1, j+1)
                alignment[i, j] += alignment[i + 1, j + 1] * move_probs[i + 1, j + 1]

            if i < n - 1:
                # Contribution from horizontal move to (i+1, j)
                alignment[i, j] += alignment[i + 1, j] * split_probs[i + 1, j]

            if j < m - 1:
                # Contribution from vertical move to (i, j+1)
                alignment[i, j] += alignment[i, j + 1] * merge_probs[i, j + 1]

    return alignment


@njit(cache=True, fastmath=True)
def _compute_soft_msm_gradient(
    x: np.ndarray,
    y: np.ndarray,
    alignment_matrix: np.ndarray,
    move_probs: np.ndarray,
    split_probs: np.ndarray,
    merge_probs: np.ndarray,
    c: float,
) -> np.ndarray:
    """
    Compute gradient considering different move types.
    """
    if x.ndim == 1:
        x = x.reshape((1, -1))
        y = y.reshape((1, -1))

    n = x.shape[1]
    m = y.shape[1]
    gradient = np.zeros(n)

    for i in range(n):
        for j in range(m):
            if alignment_matrix[i, j] > 0:
                diagonal_grad = np.sign(x[0, i] - y[0, j])
                gradient[i] += diagonal_grad * alignment_matrix[i, j] * move_probs[i, j]

                if i > 0:
                    split_grad = _msm_cost_gradient_x(x[0, i], x[0, i - 1], y[0, j], c)
                    gradient[i] += (
                        split_grad * alignment_matrix[i, j] * split_probs[i, j]
                    )
                if j > 0:
                    merge_grad = _msm_cost_gradient_y(y[0, j], x[0, i], y[0, j - 1], c)
                    gradient[i] += (
                        merge_grad * alignment_matrix[i, j] * merge_probs[i, j]
                    )

            if i < n - 1 and alignment_matrix[i + 1, j] > 0:
                split_grad_prev = _msm_cost_gradient_y(x[0, i + 1], x[0, i], y[0, j], c)
                gradient[i] += (
                    split_grad_prev * alignment_matrix[i + 1, j] * split_probs[i + 1, j]
                )

    return gradient


@njit(cache=True, fastmath=True)
def _msm_cost_gradient_x(x, y, z, c):
    """Gradient of msm_cost(x, y, z, c) w.r.t. x (first parameter)."""
    if (y <= x <= z) or (y >= x >= z):
        return 0.0
    else:
        if abs(x - y) < abs(x - z):
            return np.sign(x - y)
        elif abs(x - y) > abs(x - z):
            return np.sign(x - z)
        else:
            # Subgradient at non-differentiable point
            return 0.5 * (np.sign(x - y) + np.sign(x - z))


@njit(cache=True, fastmath=True)
def _msm_cost_gradient_y(x, y, z, c):
    """Gradient of msm_cost(x, y, z, c) w.r.t. y (second parameter)."""
    if (y <= x <= z) or (y >= x >= z):
        # When x is between y and z, changing y might change this condition
        # This requires more careful analysis of the boundary
        if abs(x - y) < 1e-10:  # x ≈ y
            return -np.sign(z - y)
        elif abs(x - z) < 1e-10:  # x ≈ z
            return np.sign(z - y)
        else:
            return 0.0
    else:
        if abs(x - y) < abs(x - z):
            return -np.sign(x - y)
        else:
            return 0.0


@njit(cache=True, fastmath=True)
def soft_msm_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: Optional[float] = None,
    c: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the Jacobian matrix of Soft-MSM distance.

    For univariate time series, this is a sparse matrix where
    J[i] = ∂d/∂x_i (i.e., a vector of gradients).

    Returns
    -------
    jacobian : np.ndarray
        Jacobian matrix of shape (len(x),)
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_jacobian(_x, _y, bounding_matrix, c, gamma)
    elif x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_jacobian(x, y, bounding_matrix, c, gamma)
    else:
        raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_msm_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    gamma: float,
    c: float,
) -> np.ndarray:
    cost_matrix, move_probs, split_probs, merge_probs = (
        _soft_msm_cost_matrix_with_probs(
            x,
            y,
            bounding_matrix=bounding_matrix,
            c=c,
            gamma=gamma,
        )
    )

    alignment_matrix = _soft_msm_alignment_matrix(
        cost_matrix, move_probs, split_probs, merge_probs
    )

    return _compute_soft_msm_gradient(
        x, y, alignment_matrix, move_probs, split_probs, merge_probs, c
    )


if __name__ == "__main__":
    # Example usage
    from aeon.testing.data_generation import make_example_2d_numpy_series

    x = make_example_2d_numpy_series(n_channels=1, n_timepoints=10, random_state=42)
    y = make_example_2d_numpy_series(n_channels=1, n_timepoints=10, random_state=43)

    gamma = 1.0
    distance = soft_msm_distance(x, y, gamma=gamma)
    print(f"Soft-MSM Distance: {distance}")

    gradient, dist = soft_msm_gradient(x, y, gamma=gamma)
    print(f"Gradient: {gradient}, Distance: {dist}")

    jacobian = soft_msm_jacobian(x, y, gamma=gamma)
    stop = ""
