"""Soft bag distance between two time series."""

__maintainer__ = []


import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._utils import _softmin2, _softmin3
from aeon.distances.pointwise._squared import _squared_dist_pointwise
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.numba._threading import threaded
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_bag_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
    alpha: float = 25.0,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_bag_distance(
            _x, _y, bounding_matrix, c=c, gamma=gamma, alpha=alpha
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_bag_distance(x, y, bounding_matrix, c=c, gamma=gamma, alpha=alpha)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_bag_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
    alpha: float = 25.0,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_bag_cost_matrix(
            _x,
            _y,
            bounding_matrix=bounding_matrix,
            c=c,
            gamma=gamma,
            alpha=alpha,
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_multivariate_bag_cost_matrix(
            x, y, bounding_matrix, c=c, gamma=gamma, alpha=alpha
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_bag_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
    alpha: float,
) -> float:
    return _soft_bag_cost_matrix(
        x,
        y,
        bounding_matrix=bounding_matrix,
        c=c,
        gamma=gamma,
        alpha=alpha,
    )[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _soft_multivariate_bag_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    distance = 0
    min_instances = min(x.shape[0], y.shape[0])
    for i in range(min_instances):
        curr_cost_matrix = _soft_bag_cost_matrix(
            x[i],
            y[i],
            bounding_matrix=bounding_matrix,
            c=c,
            gamma=gamma,
            alpha=alpha,
        )
        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
        distance += curr_cost_matrix[-1, -1]
    return cost_matrix


@njit(fastmath=True, cache=True)
def _soft_bag_transition_cost(x_val, y_prev, z_other, c, alpha, gamma):
    s = -((x_val - y_prev) * (x_val - z_other))  # >0 when between
    g = 1.0 / (1.0 + np.exp(-(alpha * s)))

    d_same_prev = _squared_dist_pointwise(
        x_val, y_prev
    )  # |x_i - x_{i-1}| or |y_j - y_{j-1}|
    d_cross = _squared_dist_pointwise(x_val, z_other)  # |x_i - y_j| or |y_j - x_i|
    return c + (1.0 - g) * _softmin2(d_same_prev, d_cross, gamma)


@njit(fastmath=True, cache=True)
def _soft_bag_cost_matrix(x, y, bounding_matrix, c, gamma, alpha):
    m = x.shape[1]
    n = y.shape[1]
    cm = np.empty((m, n), dtype=np.float64)

    # (0,0): diagonal (match) cost
    cm[0, 0] = _squared_dist_pointwise(x[0, 0], y[0, 0])

    # first column: vertical transitions (i-1,j) -> (i,j) with j = 0
    for i in range(1, m):
        if bounding_matrix[i, 0]:
            trans = _soft_bag_transition_cost(
                x_val=x[0, i],
                y_prev=x[0, i - 1],
                z_other=y[0, 0],
                c=c,
                alpha=alpha,
                gamma=gamma,
            )
            cm[i, 0] = cm[i - 1, 0] + trans

    # first row: horizontal transitions (i,j-1) -> (i,j) with i = 0
    for j in range(1, n):
        if bounding_matrix[0, j]:
            trans = _soft_bag_transition_cost(
                x_val=y[0, j],
                y_prev=y[0, j - 1],
                z_other=x[0, 0],
                c=c,
                alpha=alpha,
                gamma=gamma,
            )
            cm[0, j] = cm[0, j - 1] + trans

    # main DP
    for i in range(1, m):
        xi = x[0, i]
        xim1 = x[0, i - 1]
        for j in range(1, n):
            if bounding_matrix[i, j]:
                yj = y[0, j]
                yjm1 = y[0, j - 1]

                # diagonal (match)
                d1 = cm[i - 1, j - 1] + _squared_dist_pointwise(xi, yj)

                # vertical (insert in x / delete in y)
                c_vert = _soft_bag_transition_cost(
                    x_val=xi, y_prev=xim1, z_other=yj, c=c, alpha=alpha, gamma=gamma
                )
                d2 = cm[i - 1, j] + c_vert

                # horizontal (insert in y / delete in x)
                c_horz = _soft_bag_transition_cost(
                    x_val=yj, y_prev=yjm1, z_other=xi, c=c, alpha=alpha, gamma=gamma
                )
                d3 = cm[i, j - 1] + c_horz

                cm[i, j] = _softmin3(d1, d2, d3, gamma)

    return cm


@threaded
def soft_bag_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,
    gamma: float = 1.0,
    alpha: float = 25.0,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _soft_bag_pairwise_distance(
            _X, window, c, itakura_max_slope, unequal_length, gamma, alpha
        )

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_bag_from_multiple_to_multiple_distance(
        _X, _y, window, c, itakura_max_slope, unequal_length, gamma, alpha
    )


@njit(cache=True, fastmath=True, parallel=True)
def _soft_bag_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: float | None,
    c: float,
    itakura_max_slope: float | None,
    unequal_length: bool,
    gamma: float = 1.0,
    alpha: float = 25.0,
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
            distances[i, j] = _soft_bag_distance(x1, x2, bounding_matrix, c)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _soft_bag_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: float | None,
    c: float,
    itakura_max_slope: float | None,
    unequal_length: bool,
    gamma: float,
    alpha: float,
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
            distances[i, j] = _soft_bag_distance(
                x1, y1, bounding_matrix, c=c, gamma=gamma, alpha=alpha
            )
    return distances


@njit(cache=True, fastmath=True)
def soft_bag_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
    alpha: float = 25.0,
) -> tuple[list[tuple[int, int]], float]:
    cost_matrix = soft_bag_cost_matrix(x, y, window, c, itakura_max_slope, gamma, alpha)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )


@njit(cache=True, fastmath=True)
def soft_bag_alignment_matrix(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    c: float = 1.0,
    alpha: float = 25.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> tuple[np.ndarray, float]:

    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        cost_matrix = _soft_bag_cost_matrix(
            _x, _y, bounding_matrix=bounding_matrix, c=c, gamma=gamma, alpha=alpha
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        cost_matrix = _soft_bag_cost_matrix(
            x,
            y,
            bounding_matrix=bounding_matrix,
            c=c,
            gamma=gamma,
            alpha=alpha,
        )
    return _soft_bag_alignment_matrix_univariate(
        x=x,
        y=y,
        cm=cost_matrix,
        bm=bounding_matrix,
        c=c,
        gamma=gamma,
        alpha=alpha,
    )


@njit(cache=True, fastmath=True)
def _soft_bag_alignment_matrix_univariate(
    x: np.ndarray,  # shape (m,)
    y: np.ndarray,  # shape (n,)
    cm: np.ndarray,  # shape (m, n), float
    bm: np.ndarray,  # shape (m, n), bool
    c: float,
    gamma: float,
    alpha: float,
) -> tuple[np.ndarray, float]:
    m = x.shape[1]
    n = y.shape[1]
    distance = cm[m - 1, n - 1]

    A = np.zeros((m, n), dtype=np.float64)
    A[m - 1, n - 1] = 1.0
    inv_gamma = 1.0 / gamma

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i == m - 1 and j == n - 1:
                continue

            r_ij = cm[i, j]
            e_ij = 0.0

            # diagonal successor: (ii+1, jj+1)
            if i + 1 < m and j + 1 < n and bm[i + 1, j + 1]:
                trans_cost = _squared_dist_pointwise(x[0, i + 1], y[0, j + 1])
                r_next = cm[i + 1, j + 1]
                w = np.exp((r_next - r_ij - trans_cost) * inv_gamma)
                e_ij += A[i + 1, j + 1] * w

            # vertical successor: (ii+1, jj)
            if i + 1 < m and bm[i + 1, j]:
                trans_cost = _soft_bag_transition_cost(
                    x[0, i + 1], x[0, i], y[0, j], c=c, alpha=alpha, gamma=gamma
                )
                r_next = cm[i + 1, j]
                w = np.exp((r_next - r_ij - trans_cost) * inv_gamma)
                e_ij += A[i + 1, j] * w

            # horizontal successor: (ii, jj+1)
            if j + 1 < n and bm[i, j + 1]:
                trans_cost = _soft_bag_transition_cost(
                    y[0, j + 1], y[0, j], x[0, i], c=c, alpha=alpha, gamma=gamma
                )
                r_next = cm[i, j + 1]
                w = np.exp((r_next - r_ij - trans_cost) * inv_gamma)
                e_ij += A[i, j + 1] * w

            A[i, j] = e_ij

    return A, distance
