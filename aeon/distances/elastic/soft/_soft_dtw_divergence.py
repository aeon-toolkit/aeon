# ===================== Soft-DTW Divergence: distance & pairwise =====================
import numpy as np
from numba import njit

from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_dtw import (
    _soft_dtw_cost_matrix,
    _soft_dtw_grad_x,
    soft_dtw_distance,
    soft_dtw_pairwise_distance,
)


def soft_dtw_divergence_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> float:
    """
    Soft-DTW divergence:
        D(x,y) = s(x,y) - 0.5*s(x,x) - 0.5*s(y,y),
    where s(·,·) is your soft_dtw_distance (no DTW fallback).
    """
    s_xy = soft_dtw_distance(x, y, gamma, window, itakura_max_slope)
    s_xx = soft_dtw_distance(x, x, gamma, window, itakura_max_slope)
    s_yy = soft_dtw_distance(y, y, gamma, window, itakura_max_slope)
    return s_xy - 0.5 * s_xx - 0.5 * s_yy


def soft_dtw_divergence_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Pairwise matrix of the soft-DTW divergence:
        D[i,j] = s(X_i, Y_j) - 0.5*s(X_i,X_i) - 0.5*s(Y_j,Y_j).
    Reuses soft_dtw_pairwise_distance for cross terms and computes soft self-terms.
    """
    S = soft_dtw_pairwise_distance(X, y, gamma, window, itakura_max_slope, n_jobs)

    def _as_list(arr_or_list):
        if isinstance(arr_or_list, list):
            return arr_or_list
        if isinstance(arr_or_list, np.ndarray) and arr_or_list.ndim >= 2:
            return [arr_or_list[i] for i in range(arr_or_list.shape[0])]
        return [arr_or_list]

    if y is None:
        X_list = _as_list(X)
        s_xx = np.array(
            [
                soft_dtw_distance(xi, xi, gamma, window, itakura_max_slope)
                for xi in X_list
            ]
        )
        return S - 0.5 * s_xx[:, None] - 0.5 * s_xx[None, :]

    X_list = _as_list(X)
    Y_list = _as_list(y)
    s_xx = np.array(
        [soft_dtw_distance(xi, xi, gamma, window, itakura_max_slope) for xi in X_list]
    )
    s_yy = np.array(
        [soft_dtw_distance(yj, yj, gamma, window, itakura_max_slope) for yj in Y_list]
    )
    return S - 0.5 * s_xx[:, None] - 0.5 * s_yy[None, :]


# ===================== Soft-DTW Divergence: gradient wrt x =====================


def soft_dtw_divergence_grad_x(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """
    Public API: gradient of soft-DTW divergence wrt x.
    Returns (dx, D_value). For univariate x, dx has shape (len(x),).
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for a differentiable soft-min.")

    # Normalize to (1, T) if univariate; pass through if already (C, T)
    if x.ndim == 1 and y.ndim == 1:
        X = x.reshape((1, x.shape[0]))
        Y = y.reshape((1, y.shape[0]))
    else:
        X, Y = x, y

    dx, D_val = _soft_dtw_divergence_grad_x(X, Y, gamma, window, itakura_max_slope)

    # Match public API shape for univariate x
    if x.ndim == 1:
        return dx.ravel(), D_val
    return dx, D_val


@njit(cache=True, fastmath=True)
def _soft_dtw_divergence_grad_x(
    X: np.ndarray,  # expected shape (C, T) or (1, T)
    Y: np.ndarray,  # expected shape (C, U) or (1, U)
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """
    Private (Numba): no reshaping. Reuses _soft_dtw_grad_x to assemble the divergence
    gradient.
    Returns (dx, D_value) with dx.shape == X.shape.
    """
    # 1) grad wrt first arg of s(X, Y)
    g_xy, s_xy = _soft_dtw_grad_x(X, Y, gamma, window, itakura_max_slope)

    # 2) s(X, X): need both-args gradient -> sum of two first-arg grads with X in both
    # roles
    g_xx_first, s_xx = _soft_dtw_grad_x(X, X, gamma, window, itakura_max_slope)
    g_xx_second, _ = _soft_dtw_grad_x(X, X, gamma, window, itakura_max_slope)
    g_xx_total = g_xx_first + g_xx_second

    # Assemble divergence gradient
    dx = g_xy - 0.5 * g_xx_total

    # Divergence value: D = s_xy - 0.5*s_xx - 0.5*s_yy (compute s_yy via internal DP)
    bm_yy = create_bounding_matrix(Y.shape[1], Y.shape[1], window, itakura_max_slope)
    s_yy = _soft_dtw_cost_matrix(Y, Y, bm_yy, gamma)[Y.shape[1] - 1, Y.shape[1] - 1]

    D_val = s_xy - 0.5 * s_xx - 0.5 * s_yy
    return dx, D_val
