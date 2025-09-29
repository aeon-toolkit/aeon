# ===================== Soft-MSM Divergence: distance & pairwise =====================
import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_msm import (
    _soft_msm_cost_matrix,
    _soft_msm_grad_x,
)
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.numba._threading import threaded
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_msm_divergence_distance(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1.0,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> float:
    """
    Soft-MSM divergence:
        D(x,y) = s(x,y) - 0.5*s(x,x) - 0.5*s(y,y),
    where s(·,·) is the soft-MSM score (no MSM fallback).
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape(1, x.shape[0])
        _y = y.reshape(1, y.shape[0])
    elif x.ndim == 2 and y.ndim == 2:
        _x, _y = x, y
    else:
        raise ValueError("x and y must be 1D or 2D")

    return _soft_msm_divergence_distance(_x, _y, c, gamma, window, itakura_max_slope)


@njit(cache=True, fastmath=True)
def _soft_msm_divergence_distance(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    gamma: float,
    window: float | None,
    itakura_max_slope: float | None,
) -> float:
    # s(x,y)
    bm_xy = create_bounding_matrix(x.shape[1], y.shape[1], window, itakura_max_slope)
    s_xy = _soft_msm_cost_matrix(x, y, bm_xy, c, gamma)[x.shape[1] - 1, y.shape[1] - 1]
    # s(x,x)
    bm_xx = create_bounding_matrix(x.shape[1], x.shape[1], window, itakura_max_slope)
    s_xx = _soft_msm_cost_matrix(x, x, bm_xx, c, gamma)[x.shape[1] - 1, x.shape[1] - 1]
    # s(y,y)
    bm_yy = create_bounding_matrix(y.shape[1], y.shape[1], window, itakura_max_slope)
    s_yy = _soft_msm_cost_matrix(y, y, bm_yy, c, gamma)[y.shape[1] - 1, y.shape[1] - 1]

    return s_xy - 0.5 * s_xx - 0.5 * s_yy


# ===================== Soft-MSM Divergence: pairwise (threaded + numba) ==============
@threaded
def soft_msm_divergence_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    c: float = 1.0,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,  # kept for API parity with aeon; threading handled by @threaded
) -> np.ndarray:
    """
    Pairwise matrix of the soft-MSM divergence:
        D[i,j] = s(X_i, Y_j) - 0.5*s(X_i,X_i) - 0.5*s(Y_j,Y_j),
    computed with numba-parallel recurrences.
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, _ = _convert_collection_to_numba_list(X, "X", multivariate_conversion)

    if y is None:
        return _soft_msm_divergence_pairwise(_X, c, gamma, window, itakura_max_slope)

    _Y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
    return _soft_msm_divergence_from_multiple_to_multiple(
        _X, _Y, c, gamma, window, itakura_max_slope
    )


# -------- internals (numba) --------
@njit(cache=True, fastmath=True, parallel=True)
def _soft_msm_divergence_pairwise(
    X: NumbaList[np.ndarray],
    c: float,
    gamma: float,
    window: float | None,
    itakura_max_slope: float | None,
) -> np.ndarray:
    n = len(X)
    D = np.zeros((n, n))
    for i in prange(n):
        # diagonal
        D[i, i] = _soft_msm_divergence_distance(
            X[i], X[i], c, gamma, window, itakura_max_slope
        )
        for j in range(i + 1, n):
            Dij = _soft_msm_divergence_distance(
                X[i], X[j], c, gamma, window, itakura_max_slope
            )
            D[i, j] = Dij
            D[j, i] = Dij
    return D


@njit(cache=True, fastmath=True, parallel=True)
def _soft_msm_divergence_from_multiple_to_multiple(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    c: float,
    gamma: float,
    window: float | None,
    itakura_max_slope: float | None,
) -> np.ndarray:
    n = len(x)
    m = len(y)
    D = np.zeros((n, m))
    for i in prange(n):
        for j in range(m):
            D[i, j] = _soft_msm_divergence_distance(
                x[i], y[j], c, gamma, window, itakura_max_slope
            )
    return D


# ===================== Soft-MSM Divergence: gradient wrt x =====================
def soft_msm_divergence_grad_x(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1.0,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """
    Public API: gradient of soft-MSM divergence wrt x.
    Returns (dx, D_value). For univariate x, dx has shape (len(x),).
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for a differentiable soft minimum.")

    # Normalize to (1, T) if univariate; pass through if already (C, T)
    if x.ndim == 1 and y.ndim == 1:
        X = x.reshape((1, x.shape[0]))
        Y = y.reshape((1, y.shape[0]))
    else:
        X, Y = x, y

    dx, D_val = _soft_msm_divergence_grad_x(X, Y, c, gamma, window, itakura_max_slope)

    if x.ndim == 1:
        return dx.ravel(), D_val
    return dx, D_val


@njit(cache=True, fastmath=True)
def _soft_msm_divergence_grad_x(
    X: np.ndarray,  # expected shape (C, T) or (1, T)
    Y: np.ndarray,  # expected shape (C, U) or (1, U)
    c: float = 1.0,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """
    Private (Numba): no reshaping. Reuses _soft_msm_grad_x to assemble the divergence
    gradient.
    Returns (dx, D_value) with dx.shape == X.shape.
    """
    # 1) grad wrt first arg of s(X, Y)
    g_xy, s_xy = _soft_msm_grad_x(X, Y, c, gamma, window, itakura_max_slope)

    g_xx_first, s_xx = _soft_msm_grad_x(X, X, c, gamma, window, itakura_max_slope)
    g_xx_second, _ = _soft_msm_grad_x(X, X, c, gamma, window, itakura_max_slope)
    g_xx_total = g_xx_first + g_xx_second

    # Assemble divergence gradient
    dx = g_xy - 0.5 * g_xx_total

    # Divergence value: D = s_xy - 0.5*s_xx - 0.5*s_yy
    bm_yy = create_bounding_matrix(Y.shape[1], Y.shape[1], window, itakura_max_slope)
    s_yy = _soft_msm_cost_matrix(Y, Y, bm_yy, c, gamma)[Y.shape[1] - 1, Y.shape[1] - 1]

    D_val = s_xy - 0.5 * s_xx - 0.5 * s_yy
    return dx, D_val
