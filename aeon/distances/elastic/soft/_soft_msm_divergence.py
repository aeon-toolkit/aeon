# ----------------------- soft-MSM divergence API -----------------------
import numpy as np
from numba import njit

from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._soft_msm import (
    _soft_msm_cost_matrix,
    _soft_msm_grad_x,
    soft_msm_distance,
    soft_msm_pairwise_distance,
)


def soft_msm_divergence_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
) -> float:
    """
    Soft-MSM divergence:
        D(x,y) = s(x,y) - 0.5*s(x,x) - 0.5*s(y,y)
    """
    s_xy = soft_msm_distance(x, y, window, c, itakura_max_slope, gamma)
    s_xx = soft_msm_distance(x, x, window, c, itakura_max_slope, gamma)
    s_yy = soft_msm_distance(y, y, window, c, itakura_max_slope, gamma)
    return s_xy - 0.5 * s_xx - 0.5 * s_yy


def soft_msm_divergence_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Pairwise matrix of the soft-MSM divergence.
    Efficiently computes:
        D[i,j] = s(X_i, Y_j) - 0.5*s(X_i,X_i) - 0.5*s(Y_j,Y_j)
    by reusing the existing pairwise soft-MSM and diagonal self-terms.
    """
    # base soft-MSM matrix
    S = soft_msm_pairwise_distance(X, y, window, c, itakura_max_slope, n_jobs, gamma)

    # self terms for rows
    if y is None:
        # to-self case
        X_list = X if isinstance(X, list) else [X[i] for i in range(len(X))]
        s_xx = np.array(
            [
                soft_msm_distance(xi, xi, window, c, itakura_max_slope, gamma)
                for xi in X_list
            ]
        )
        # broadcast: D = S - 0.5*s_xx[:,None] - 0.5*s_xx[None,:]
        return S - 0.5 * s_xx[:, None] - 0.5 * s_xx[None, :]

    # to-other case
    X_list = X if isinstance(X, list) else [X[i] for i in range(len(X))]
    Y_list = y if isinstance(y, list) else [y[i] for i in range(len(y))]
    s_xx = np.array(
        [
            soft_msm_distance(xi, xi, window, c, itakura_max_slope, gamma)
            for xi in X_list
        ]
    )
    s_yy = np.array(
        [
            soft_msm_distance(yi, yi, window, c, itakura_max_slope, gamma)
            for yi in Y_list
        ]
    )
    return S - 0.5 * s_xx[:, None] - 0.5 * s_yy[None, :]


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

    # Match public API shape for univariate x
    if x.ndim == 1:
        return dx.ravel(), D_val
    return dx, D_val


@njit(cache=True, fastmath=True)
def _soft_msm_divergence_grad_x(
    X: np.ndarray,  # expected shape (1, T) for univariate (no reshaping here)
    Y: np.ndarray,  # expected shape (1, U)
    c: float = 1.0,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """
    Private (Numba): no reshaping. Reuses _soft_msm_grad_x to assemble the divergence
    gradient.
    Returns (dx, D_value), where dx has shape (T,) (same as _soft_msm_grad_x output).
    """
    # 1) grad wrt first arg of s(X, Y)
    g_xy, s_xy = _soft_msm_grad_x(X, Y, c, gamma, window, itakura_max_slope)

    # 2) s(X, X): need both-args gradient -> sum of two first-arg grads with X in both
    # roles
    g_xx_first, s_xx = _soft_msm_grad_x(X, X, c, gamma, window, itakura_max_slope)
    g_xx_second, _ = _soft_msm_grad_x(X, X, c, gamma, window, itakura_max_slope)
    g_xx_total = g_xx_first + g_xx_second

    # Assemble divergence gradient
    dx = g_xy - 0.5 * g_xx_total

    # Divergence value: D = s_xy - 0.5*s_xx - 0.5*s_yy
    # Compute s_yy with internal soft MSM DP to stay in Numba (no Python calls)
    bm_yy = create_bounding_matrix(Y.shape[1], Y.shape[1], window, itakura_max_slope)
    s_yy = _soft_msm_cost_matrix(Y, Y, bm_yy, c, gamma)[Y.shape[1] - 1, Y.shape[1] - 1]

    D_val = s_xy - 0.5 * s_xx - 0.5 * s_yy
    return dx, D_val
