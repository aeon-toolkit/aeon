# ===================== Soft-DTW Divergence: distance & pairwise =====================
import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_dtw import (
    _soft_dtw_cost_matrix,
    _soft_dtw_distance,
    _soft_dtw_grad_x,
)
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.numba._threading import threaded
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_dtw_divergence_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape(1, x.shape[0])
        _y = y.reshape(1, y.shape[0])
    elif x.ndim == 2 and y.ndim == 2:
        _x, _y = x, y
    else:
        raise ValueError("x and y must be 1D or 2D")

    return _soft_dtw_divergence_distance(_x, _y, gamma, window, itakura_max_slope)


@njit(cache=True, fastmath=True)
def _soft_dtw_divergence_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    window: float | None,
    itakura_max_slope: float | None,
) -> float:
    bm_xy = create_bounding_matrix(x.shape[1], y.shape[1], window, itakura_max_slope)
    s_xy = _soft_dtw_distance(x, y, bm_xy, gamma)
    bm_xx = create_bounding_matrix(x.shape[1], x.shape[1], window, itakura_max_slope)
    s_xx = _soft_dtw_distance(x, x, bm_xx, gamma)
    bm_yy = create_bounding_matrix(y.shape[1], y.shape[1], window, itakura_max_slope)
    s_yy = _soft_dtw_distance(y, y, bm_yy, gamma)

    return s_xy - 0.5 * s_xx - 0.5 * s_yy


# ===================== Soft-DTW Divergence: pairwise (threaded + numba) ==============
@threaded
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
        D[i,j] = s(X_i, Y_j) - 0.5*s(X_i,X_i) - 0.5*s(Y_j,Y_j),
    computed with numba-parallel recurrences.

    Parameters
    ----------
    X : np.ndarray or list[np.ndarray]
        Collection of time series (univariate shape (T,) or multivariate (C,T)).
    y : np.ndarray or list[np.ndarray] or None, default=None
        Optional second collection. If None, compute a square (to-self) matrix.
    gamma : float, default=1.0
        Soft-min temperature.
    window : float or None, default=None
        Sakoe–Chiba band width (fraction of length) or None.
    itakura_max_slope : float or None, default=None
        Itakura parallelogram slope bound.
    n_jobs : int, default=1
        Parallelism for the outer Python wrapper (same as aeon’s @threaded).

    Returns
    -------
    np.ndarray
        Pairwise soft-DTW divergence matrix.
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, _ = _convert_collection_to_numba_list(X, "X", multivariate_conversion)

    if y is None:
        return _soft_dtw_divergence_pairwise(_X, window, itakura_max_slope, gamma)

    _Y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
    # use the more general cross variant (handles unequal lengths independently)
    return _soft_dtw_divergence_from_multiple_to_multiple(
        _X, _Y, window, itakura_max_slope, gamma
    )


# -------- internals (numba) --------
@njit(cache=True, fastmath=True, parallel=True)
def _soft_dtw_divergence_pairwise(
    X: NumbaList[np.ndarray],
    window: float | None,
    itakura_max_slope: float | None,
    gamma: float,
) -> np.ndarray:
    n = len(X)
    D = np.zeros((n, n))
    for i in prange(n):
        # compute diagonal once
        D[i, i] = _soft_dtw_divergence_distance(
            X[i], X[i], gamma, window, itakura_max_slope
        )
        for j in range(i + 1, n):
            Dij = _soft_dtw_divergence_distance(
                X[i], X[j], gamma, window, itakura_max_slope
            )
            D[i, j] = Dij
            D[j, i] = Dij
    return D


@njit(cache=True, fastmath=True, parallel=True)
def _soft_dtw_divergence_from_multiple_to_multiple(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: float | None,
    itakura_max_slope: float | None,
    gamma: float,
) -> np.ndarray:
    n = len(x)
    m = len(y)
    D = np.zeros((n, m))

    for i in prange(n):
        for j in range(m):
            D[i, j] = _soft_dtw_divergence_distance(
                x[i], y[j], gamma, window, itakura_max_slope
            )
    return D


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
