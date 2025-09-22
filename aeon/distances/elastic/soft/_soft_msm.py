"""Move-split-merge (soft_msm) distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._utils import _softmin2, _softmin3
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.numba._threading import threaded
from aeon.utils.validation.collection import _is_numpy_list_multivariate


# ---------------------------------- public API ----------------------------------

@njit(cache=True, fastmath=True)
def soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(_x, _y, bounding_matrix, c=c, gamma=gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(x, y, bounding_matrix, c=c, gamma=gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_cost_matrix(_x, _y, bounding_matrix=bounding_matrix, c=c, gamma=gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_cost_matrix(x, y, bounding_matrix, c=c, gamma=gamma)
    raise ValueError("x and y must be 1D or 2D")


@threaded
def soft_msm_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,
    gamma: float = 1.0,
) -> np.ndarray:
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _soft_msm_pairwise_distance(
            _X, window, c, itakura_max_slope, unequal_length, gamma
        )

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_msm_from_multiple_to_multiple_distance(
        _X, _y, window, c, itakura_max_slope, unequal_length, gamma
    )


@njit(cache=True, fastmath=True)
def soft_msm_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
) -> tuple[list[tuple[int, int]], float]:
    cost_matrix = soft_msm_cost_matrix(x, y, window, c, itakura_max_slope, gamma)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )


@njit(cache=True, fastmath=True)
def soft_msm_alignment_matrix(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    c: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> tuple[np.ndarray, float]:
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bm = create_bounding_matrix(_x.shape[1], _y.shape[1], window, itakura_max_slope)
        cm = _soft_msm_cost_matrix(_x, _y, bounding_matrix=bm, c=c, gamma=gamma)
        return _soft_msm_alignment_matrix_univariate(_x, _y, cm, bm, c, gamma)
    else:
        bm = create_bounding_matrix(x.shape[1], y.shape[1], window, itakura_max_slope)
        cm = _soft_msm_cost_matrix(x, y, bounding_matrix=bm, c=c, gamma=gamma)
        return _soft_msm_alignment_matrix_univariate(x, y, cm, bm, c, gamma)


def soft_msm_grad_x(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1.0,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """
    Gradient (Jacobian) of soft-MSM distance w.r.t. the univariate series x.
    Returns (dx, distance) with dx shape (len(x),).
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for a differentiable soft minimum.")
    if x.ndim == 1 and y.ndim == 1:
        X = x.reshape((1, x.shape[0]))
        Y = y.reshape((1, y.shape[0]))
    else:
        X = x
        Y = y
    return _soft_msm_grad_x(X, Y, c, gamma, window, itakura_max_slope)


# --------------------------------- internals ---------------------------------

@njit(cache=True, fastmath=True)
def _soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
) -> float:
    return _soft_msm_cost_matrix(
        x, y, bounding_matrix=bounding_matrix, c=c, gamma=gamma
    )[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _soft_multivariate_msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    min_instances = min(x.shape[0], y.shape[0])
    for i in range(min_instances):
        curr_cost_matrix = _soft_msm_cost_matrix(
            x[i], y[i], bounding_matrix=bounding_matrix, c=c, gamma=gamma
        )
        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
    return cost_matrix


@njit(fastmath=True, cache=True)
def _soft_msm_transition_cost(x_val, y_prev, z_other, c, gamma):
    a = x_val - y_prev
    b = x_val - z_other

    # Between gate
    u = a * b
    g = 0.5 * (1.0 - u / np.sqrt(u * u + 1e-9))

    d_same_prev = a * a
    d_cross = b * b
    return c + (1.0 - g) * _softmin2(d_same_prev, d_cross, gamma)


@njit(fastmath=True, cache=True)
def _soft_msm_cost_matrix(x, y, bounding_matrix, c, gamma):
    m = x.shape[1]
    n = y.shape[1]
    cm = np.full((m, n), np.inf, dtype=np.float64)

    cm[0, 0] = (x[0, 0] - y[0, 0]) ** 2

    # first column: vertical transitions (i-1,j) -> (i,j) with j = 0
    for i in range(1, m):
        if bounding_matrix[i, 0]:
            trans = _soft_msm_transition_cost(
                x_val=x[0, i],
                y_prev=x[0, i - 1],
                z_other=y[0, 0],
                c=c,
                gamma=gamma,
            )
            cm[i, 0] = cm[i - 1, 0] + trans

    # first row: horizontal transitions (i,j-1) -> (i,j) with i = 0
    for j in range(1, n):
        if bounding_matrix[0, j]:
            trans = _soft_msm_transition_cost(
                x_val=y[0, j],
                y_prev=y[0, j - 1],
                z_other=x[0, 0],
                c=c,
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
                d1 = cm[i - 1, j - 1] + (xi - yj) ** 2

                # vertical (insert in x / delete in y)
                c_vert = _soft_msm_transition_cost(
                    x_val=xi, y_prev=xim1, z_other=yj, c=c, gamma=gamma
                )
                d2 = cm[i - 1, j] + c_vert

                # horizontal (insert in y / delete in x)
                c_horz = _soft_msm_transition_cost(
                    x_val=yj, y_prev=yjm1, z_other=xi, c=c, gamma=gamma
                )
                d3 = cm[i, j - 1] + c_horz

                cm[i, j] = _softmin3(d1, d2, d3, gamma)

    return cm


@njit(cache=True, fastmath=True, parallel=True)
def _soft_msm_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: float | None,
    c: float,
    itakura_max_slope: float | None,
    unequal_length: bool,
    gamma: float = 1.0,
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
            distances[i, j] = _soft_msm_distance(
                x1, x2, bounding_matrix=bounding_matrix, c=c, gamma=gamma
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _soft_msm_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: float | None,
    c: float,
    itakura_max_slope: float | None,
    unequal_length: bool,
    gamma: float,
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
            distances[i, j] = _soft_msm_distance(
                x1, y1, bounding_matrix, c=c, gamma=gamma
            )
    return distances


@njit(cache=True, fastmath=True)
def _soft_msm_alignment_matrix_univariate(
    x: np.ndarray,  # shape (1, m)
    y: np.ndarray,  # shape (1, n)
    cm: np.ndarray,  # shape (m, n)
    bm: np.ndarray,  # shape (m, n)
    c: float,
    gamma: float,
) -> tuple[np.ndarray, float]:
    m = x.shape[1]
    n = y.shape[1]
    distance = cm[m - 1, n - 1]

    inv_gamma = 1.0 / gamma

    # Backward node adjoint (∂R_goal/∂R[i,j])
    A = np.zeros((m, n), dtype=np.float64)
    A[m - 1, n - 1] = 1.0

    # Diagonal-edge occupancy (∂R_goal/∂((x_i - y_j)^2))
    G = np.zeros((m, n), dtype=np.float64)

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i == m - 1 and j == n - 1:
                continue

            r_ij = cm[i, j]
            if (not bm[i, j]) or (not np.isfinite(r_ij)):
                continue

            acc = 0.0

            # (i,j) -> (i+1, j+1): diagonal / match
            if (i + 1 < m) and (j + 1 < n) and bm[i + 1, j + 1]:
                match_cost = (x[0, i + 1] - y[0, j + 1]) ** 2
                r_next = cm[i + 1, j + 1]
                w_diag = np.exp((r_next - (r_ij + match_cost)) * inv_gamma)
                acc += A[i + 1, j + 1] * w_diag
                G[i + 1, j + 1] += A[i + 1, j + 1] * w_diag

            # (i,j) -> (i+1, j): vertical
            if (i + 1 < m) and bm[i + 1, j]:
                trans_cost = _soft_msm_transition_cost(
                    x[0, i + 1], x[0, i], y[0, j], c=c, gamma=gamma
                )
                r_next = cm[i + 1, j]
                w_vert = np.exp((r_next - (r_ij + trans_cost)) * inv_gamma)
                acc += A[i + 1, j] * w_vert

            # (i,j) -> (i, j+1): horizontal
            if (j + 1 < n) and bm[i, j + 1]:
                trans_cost = _soft_msm_transition_cost(
                    y[0, j + 1], y[0, j], x[0, i], c=c, gamma=gamma
                )
                r_next = cm[i, j + 1]
                w_horz = np.exp((r_next - (r_ij + trans_cost)) * inv_gamma)
                acc += A[i, j + 1] * w_horz

            A[i, j] = acc

    # base match cell contribution is weighted by A[0,0]
    G[0, 0] += A[0, 0]

    return G, distance


@njit(cache=True, fastmath=True)
def _soft_msm_transition_cost_grads(x_val, y_prev, z_other, c, gamma):
    """
    Derivatives of transition cost wrt (x_val, y_prev, z_other).
    trans = c + (1 - g) * softmin( (x_val - y_prev)^2, (x_val - z_other)^2 )
    with g = 0.5 * (1 - (ab)/sqrt((ab)^2+eps)), a=x_val-y_prev, b=x_val-z_other.
    """
    a = x_val - y_prev
    b = x_val - z_other

    # softmin bits for the (d_same_prev, d_cross)
    t1 = -(a * a) / gamma
    t2 = -(b * b) / gamma
    tmax = max(t1, t2)
    e1 = np.exp(t1 - tmax)
    e2 = np.exp(t2 - tmax)
    Z = e1 + e2
    w_same = e1 / Z
    w_cross = e2 / Z
    # softmin value
    softmin_val = -gamma * (np.log(Z) + tmax)

    # gate g and its gradients (parameter-free)
    eps = 1e-9
    u = a * b
    denom_sq = u * u + eps
    denom = np.sqrt(denom_sq)
    g = 0.5 * (1.0 - u / denom)

    # df/du for f(u) = u/sqrt(u^2+eps) equals eps/(u^2+eps)^(3/2)
    df_du = eps / (denom_sq * denom)
    dg_du = -0.5 * df_du

    # du/dargs
    du_dxval = a + b
    du_dyprev = -b
    du_dzoth = -a

    dg_dxval = dg_du * du_dxval
    dg_dyprev = dg_du * du_dyprev
    dg_dzoth = dg_du * du_dzoth

    # ∂softmin/∂args
    dsm_dxval = w_same * (2.0 * a) + w_cross * (2.0 * b)
    dsm_dyprev = w_same * (-2.0 * a)
    dsm_dzoth = w_cross * (-2.0 * b)

    # chain rule: d trans = -(dg)*softmin + (1-g)*d softmin
    dtrans_dxval = -dg_dxval * softmin_val + (1.0 - g) * dsm_dxval
    dtrans_dyprev = -dg_dyprev * softmin_val + (1.0 - g) * dsm_dyprev
    dtrans_dzoth = -dg_dzoth * softmin_val + (1.0 - g) * dsm_dzoth

    return dtrans_dxval, dtrans_dyprev, dtrans_dzoth


@njit(cache=True, fastmath=True)
def _soft_msm_grad_x(
    X: np.ndarray,
    Y: np.ndarray,
    c: float = 1.0,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    m, n = X.shape[1], Y.shape[1]

    # bounding + forward DP costs
    bm = create_bounding_matrix(m, n, window, itakura_max_slope)
    R = _soft_msm_cost_matrix(X, Y, bm, c, gamma)  # shape (m, n)

    # Conditional move probabilities from each node (softmin weights)
    w_diag = np.zeros((m, n))
    w_vert = np.zeros((m, n))
    w_horz = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            Rij = R[i, j]
            if (not bm[i, j]) or (not np.isfinite(Rij)):
                continue
            # diagonal / match to (i+1, j+1)
            if i + 1 < m and j + 1 < n and bm[i + 1, j + 1]:
                match_cost = (X[0, i + 1] - Y[0, j + 1]) ** 2
                w_diag[i, j] = np.exp((R[i + 1, j + 1] - (Rij + match_cost)) / gamma)
            # vertical to (i+1, j)
            if i + 1 < m and bm[i + 1, j]:
                trans_cost = _soft_msm_transition_cost(
                    X[0, i + 1], X[0, i], Y[0, j], c=c, gamma=gamma
                )
                w_vert[i, j] = np.exp((R[i + 1, j] - (Rij + trans_cost)) / gamma)
            # horizontal to (i, j+1)
            if j + 1 < n and bm[i, j + 1]:
                trans_cost = _soft_msm_transition_cost(
                    Y[0, j + 1], Y[0, j], X[0, i], c=c, gamma=gamma
                )
                w_horz[i, j] = np.exp((R[i, j + 1] - (Rij + trans_cost)) / gamma)

    # Backward node occupancy A (node adjoint)
    A = np.zeros((m, n))
    A[m - 1, n - 1] = 1.0
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i == m - 1 and j == n - 1:
                continue
            if (not bm[i, j]) or (not np.isfinite(R[i, j])):
                continue
            acc = 0.0
            if i + 1 < m and j + 1 < n and bm[i + 1, j + 1]:
                acc += A[i + 1, j + 1] * w_diag[i, j]
            if i + 1 < m and bm[i + 1, j]:
                acc += A[i + 1, j] * w_vert[i, j]
            if j + 1 < n and bm[i, j + 1]:
                acc += A[i, j + 1] * w_horz[i, j]
            A[i, j] = acc

    # Edge occupancies (use A[child] * w_edge(parent))
    Gdiag = np.zeros((m, n))  # diagonal edge into (i,j)
    Vocc = np.zeros((m, n))   # vertical edge into (i,j)
    Hocc = np.zeros((m, n))   # horizontal edge into (i,j)

    for i in range(m):
        for j in range(n):
            if (not bm[i, j]) or (not np.isfinite(R[i, j])):
                continue
            if i + 1 < m and j + 1 < n and bm[i + 1, j + 1]:
                Gdiag[i + 1, j + 1] += A[i + 1, j + 1] * w_diag[i, j]
            if i + 1 < m and bm[i + 1, j]:
                Vocc[i + 1, j] += A[i + 1, j] * w_vert[i, j]
            if j + 1 < n and bm[i, j + 1]:
                Hocc[i, j + 1] += A[i, j + 1] * w_horz[i, j]

    # Gradient wrt x (shape m,)
    dx = np.zeros(m)

    # (i) match costs (diagonal edges) + base cell
    if bm[0, 0]:
        dx[0] += 2.0 * (X[0, 0] - Y[0, 0]) * A[0, 0]
    for i in range(1, m):
        for j in range(1, n):
            if Gdiag[i, j] != 0.0:
                dx[i] += 2.0 * (X[0, i] - Y[0, j]) * Gdiag[i, j]

    # (ii) vertical transitions: edge (i-1,j) -> (i,j)
    for i in range(1, m):
        for j in range(n):
            occ = Vocc[i, j]
            if occ == 0.0:
                continue
            d_dxval, d_dyprev, _ = _soft_msm_transition_cost_grads(
                X[0, i], X[0, i - 1], Y[0, j], c=c, gamma=gamma
            )
            dx[i] += occ * d_dxval      # x_val is X[i]
            dx[i - 1] += occ * d_dyprev # y_prev is X[i-1]

    # (iii) horizontal transitions: edge (i,j-1) -> (i,j); x appears as z_other
    for i in range(m):
        for j in range(1, n):
            occ = Hocc[i, j]
            if occ == 0.0:
                continue
            # trans args: (x_val = Y[j], y_prev = Y[j-1], z_other = X[i])
            _, _, d_dzoth = _soft_msm_transition_cost_grads(
                Y[0, j], Y[0, j - 1], X[0, i], c=c, gamma=gamma
            )
            dx[i] += occ * d_dzoth

    return dx, R[-1, -1]

if __name__ == "__main__":
    from aeon.testing.data_generation import make_example_2d_numpy_series
    n_timepoints = 10000

    x = make_example_2d_numpy_series(n_timepoints, 1, random_state=1)
    y = make_example_2d_numpy_series(n_timepoints, 1, random_state=2)

    from time import time
    t0 = time()
    distances = soft_msm_grad_x(x, y, c=1.0, gamma=1.0)
    print(f"Time: {time()-t0:.3f}s")
