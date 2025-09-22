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


@njit(cache=True, fastmath=True)
def soft_msm_distance(
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
        return _soft_msm_distance(
            _x, _y, bounding_matrix, c=c, gamma=gamma, alpha=alpha
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_distance(x, y, bounding_matrix, c=c, gamma=gamma, alpha=alpha)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_msm_cost_matrix(
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
        return _soft_msm_cost_matrix(
            _x, _y, bounding_matrix=bounding_matrix, c=c, gamma=gamma, alpha=alpha
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_multivariate_msm_cost_matrix(
            x, y, bounding_matrix, c=c, gamma=gamma, alpha=alpha
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
    alpha: float,
) -> float:
    return _soft_msm_cost_matrix(
        x, y, bounding_matrix=bounding_matrix, c=c, gamma=gamma, alpha=alpha
    )[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _soft_multivariate_msm_cost_matrix(
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
    min_instances = min(x.shape[0], y.shape[0])
    for i in range(min_instances):
        curr_cost_matrix = _soft_msm_cost_matrix(
            x[i], y[i], bounding_matrix=bounding_matrix, c=c, gamma=gamma, alpha=alpha
        )
        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
    return cost_matrix


@njit(fastmath=True, cache=True)
def _soft_msm_transition_cost(x_val, y_prev, z_other, c, alpha, gamma):
    s = -((x_val - y_prev) * (x_val - z_other))  # >0 when between
    g = 1.0 / (1.0 + np.exp(-(alpha * s)))

    d_same_prev = (x_val - y_prev) ** 2
    d_cross = (x_val - z_other) ** 2
    return c + (1.0 - g) * _softmin2(d_same_prev, d_cross, gamma)


@njit(fastmath=True, cache=True)
def _soft_msm_cost_matrix(x, y, bounding_matrix, c, gamma, alpha):
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
                alpha=alpha,
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
                d1 = cm[i - 1, j - 1] + (xi - yj) ** 2

                # vertical (insert in x / delete in y)
                c_vert = _soft_msm_transition_cost(
                    x_val=xi, y_prev=xim1, z_other=yj, c=c, alpha=alpha, gamma=gamma
                )
                d2 = cm[i - 1, j] + c_vert

                # horizontal (insert in y / delete in x)
                c_horz = _soft_msm_transition_cost(
                    x_val=yj, y_prev=yjm1, z_other=xi, c=c, alpha=alpha, gamma=gamma
                )
                d3 = cm[i, j - 1] + c_horz

                cm[i, j] = _softmin3(d1, d2, d3, gamma)

    return cm


@threaded
def soft_msm_pairwise_distance(
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
        return _soft_msm_pairwise_distance(
            _X, window, c, itakura_max_slope, unequal_length, gamma, alpha
        )

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_msm_from_multiple_to_multiple_distance(
        _X, _y, window, c, itakura_max_slope, unequal_length, gamma, alpha
    )


@njit(cache=True, fastmath=True, parallel=True)
def _soft_msm_pairwise_distance(
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
            distances[i, j] = _soft_msm_distance(
                x1, x2, bounding_matrix=bounding_matrix, c=c, gamma=gamma, alpha=alpha
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
            distances[i, j] = _soft_msm_distance(
                x1, y1, bounding_matrix, c=c, gamma=gamma, alpha=alpha
            )
    return distances


@njit(cache=True, fastmath=True)
def soft_msm_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
    alpha: float = 25.0,
) -> tuple[list[tuple[int, int]], float]:
    cost_matrix = soft_msm_cost_matrix(x, y, window, c, itakura_max_slope, gamma, alpha)
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
        cost_matrix = _soft_msm_cost_matrix(
            _x, _y, bounding_matrix=bounding_matrix, c=c, gamma=gamma, alpha=alpha
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        cost_matrix = _soft_msm_cost_matrix(
            x,
            y,
            bounding_matrix=bounding_matrix,
            c=c,
            gamma=gamma,
            alpha=alpha,
        )
    return _soft_msm_alignment_matrix_univariate(
        x=x,
        y=y,
        cm=cost_matrix,
        bm=bounding_matrix,
        c=c,
        gamma=gamma,
        alpha=alpha,
    )


@njit(cache=True, fastmath=True)
def _soft_msm_alignment_matrix_univariate(
    x: np.ndarray,  # shape (1, m)
    y: np.ndarray,  # shape (1, n)
    cm: np.ndarray,  # shape (m, n)
    bm: np.ndarray,  # shape (m, n)
    c: float,
    gamma: float,
    alpha: float,
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
            acc = 0.0

            # (i,j) -> (i+1, j+1): diagonal / match
            if (i + 1 < m) and (j + 1 < n) and bm[i + 1, j + 1]:
                match_cost = (x[0, i + 1] - y[0, j + 1]) ** 2
                r_next = cm[i + 1, j + 1]
                w_diag = np.exp((r_next - r_ij - match_cost) * inv_gamma)
                acc += A[i + 1, j + 1] * w_diag
                G[i + 1, j + 1] += A[i + 1, j + 1] * w_diag

            # (i,j) -> (i+1, j): vertical
            if (i + 1 < m) and bm[i + 1, j]:
                trans_cost = _soft_msm_transition_cost(
                    x[0, i + 1], x[0, i], y[0, j], c=c, alpha=alpha, gamma=gamma
                )
                r_next = cm[i + 1, j]
                w_vert = np.exp((r_next - r_ij - trans_cost) * inv_gamma)
                acc += A[i + 1, j] * w_vert

            # (i,j) -> (i, j+1): horizontal
            if (j + 1 < n) and bm[i, j + 1]:
                trans_cost = _soft_msm_transition_cost(
                    y[0, j + 1], y[0, j], x[0, i], c=c, alpha=alpha, gamma=gamma
                )
                r_next = cm[i, j + 1]
                w_horz = np.exp((r_next - r_ij - trans_cost) * inv_gamma)
                acc += A[i, j + 1] * w_horz

            A[i, j] = acc

    return G, distance


@njit(cache=True, fastmath=True)
def _soft_msm_transition_cost_grads(x_val, y_prev, z_other, c, alpha, gamma):
    """
    Derivatives of your transition cost wrt (x_val, y_prev, z_other).
    trans = c + (1 - g) * softmin( (x_val - y_prev)^2, (x_val - z_other)^2 )
    with g = sigmoid(alpha * s), s = - (x_val - y_prev)*(x_val - z_other)
    """
    a = x_val - y_prev
    b = x_val - z_other

    # softmin bits
    t1 = -(a * a) / gamma
    t2 = -(b * b) / gamma
    tmax = max(t1, t2)
    e1 = np.exp(t1 - tmax)
    e2 = np.exp(t2 - tmax)
    Z = e1 + e2
    w_same = e1 / Z
    w_cross = e2 / Z
    # softmin value (needed because d/d(1-g) multiplies softmin)
    softmin_val = -gamma * (np.log(Z) + tmax)

    # g and its derivatives
    s = -a * b
    g = 1.0 / (1.0 + np.exp(-alpha * s))
    g_fac = alpha * g * (1.0 - g)

    # ∂s/∂args
    ds_dxval = -(a + b)  # = -(2*x_val - y_prev - z_other)
    ds_dyprev = b  # =  (x_val - z_other)
    ds_dzoth = a  # =  (x_val - y_prev)

    # ∂softmin/∂args
    dsm_dxval = w_same * (2.0 * a) + w_cross * (2.0 * b)
    dsm_dyprev = w_same * (-2.0 * a)
    dsm_dzoth = w_cross * (-2.0 * b)

    # chain rule: d trans = -(dg)*softmin + (1-g)*d softmin
    dtrans_dxval = -g_fac * ds_dxval * softmin_val + (1.0 - g) * dsm_dxval
    dtrans_dyprev = -g_fac * ds_dyprev * softmin_val + (1.0 - g) * dsm_dyprev
    dtrans_dzoth = -g_fac * ds_dzoth * softmin_val + (1.0 - g) * dsm_dzoth

    return dtrans_dxval, dtrans_dyprev, dtrans_dzoth


def soft_msm_grad_x(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1.0,
    gamma: float = 1.0,
    alpha: float = 25.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """
    Gradient (Jacobian) of soft-MSM distance w.r.t. the univariate series x.
    Returns a vector of shape (len(x),).

    Requires your helpers in scope:
      - create_bounding_matrix(...)
      - _soft_msm_cost_matrix(...)
      - _soft_msm_transition_cost(...)
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for a differentiable soft minimum.")
    # ensure univariate (1, m) / (1, n)
    if x.ndim == 1 and y.ndim == 1:
        X = x.reshape((1, x.shape[0]))
        Y = y.reshape((1, y.shape[0]))
    else:
        X = x
        Y = y

    return _soft_msm_grad_x(X, Y, c, gamma, alpha, window, itakura_max_slope)


@njit(cache=True, fastmath=True)
def _soft_msm_grad_x(
    X: np.ndarray,
    Y: np.ndarray,
    c: float = 1.0,
    gamma: float = 1.0,
    alpha: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    m, n = X.shape[1], Y.shape[1]

    # bounding + forward DP costs
    bm = create_bounding_matrix(m, n, window, itakura_max_slope)
    R = _soft_msm_cost_matrix(X, Y, bm, c, gamma, alpha)  # shape (m, n)

    # Precompute conditional move probabilities w.r.t. each node (i,j)
    # w_* sum to 1 for valid outgoing moves
    w_diag = np.zeros((m, n))
    w_vert = np.zeros((m, n))
    w_horz = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            Rij = R[i, j]
            # diagonal / match to (i+1, j+1)
            if i + 1 < m and j + 1 < n and bm[i + 1, j + 1]:
                match_cost = (X[0, i + 1] - Y[0, j + 1]) ** 2
                w_diag[i, j] = np.exp((R[i + 1, j + 1] - Rij - match_cost) / gamma)
            # vertical to (i+1, j)
            if i + 1 < m and bm[i + 1, j]:
                trans_cost = _soft_msm_transition_cost(
                    X[0, i + 1], X[0, i], Y[0, j], c=c, alpha=alpha, gamma=gamma
                )
                w_vert[i, j] = np.exp((R[i + 1, j] - Rij - trans_cost) / gamma)
            # horizontal to (i, j+1)
            if j + 1 < n and bm[i, j + 1]:
                trans_cost = _soft_msm_transition_cost(
                    Y[0, j + 1], Y[0, j], X[0, i], c=c, alpha=alpha, gamma=gamma
                )
                w_horz[i, j] = np.exp((R[i, j + 1] - Rij - trans_cost) / gamma)

    # Backward node occupancy A (equals visitation probability μ)
    A = np.zeros((m, n))
    A[m - 1, n - 1] = 1.0
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i == m - 1 and j == n - 1:
                continue
            acc = 0.0
            if i + 1 < m and j + 1 < n and bm[i + 1, j + 1]:
                acc += A[i + 1, j + 1] * w_diag[i, j]
            if i + 1 < m and bm[i + 1, j]:
                acc += A[i + 1, j] * w_vert[i, j]
            if j + 1 < n and bm[i, j + 1]:
                acc += A[i, j + 1] * w_horz[i, j]
            A[i, j] = acc

    # Edge occupancies (using A[current] * w_edge)
    Gdiag = np.zeros((m, n))  # diagonal edge into (i,j)
    Vocc = np.zeros((m, n))  # vertical edge into (i,j)
    Hocc = np.zeros((m, n))  # horizontal edge into (i,j)

    for i in range(m):
        for j in range(n):
            if i + 1 < m and j + 1 < n and bm[i + 1, j + 1]:
                Gdiag[i + 1, j + 1] += A[i, j] * w_diag[i, j]
            if i + 1 < m and bm[i + 1, j]:
                Vocc[i + 1, j] += A[i, j] * w_vert[i, j]
            if j + 1 < n and bm[i, j + 1]:
                Hocc[i, j + 1] += A[i, j] * w_horz[i, j]

    # Gradient wrt x (shape m,)
    dx = np.zeros(m)

    # (i) match costs (diagonal edges) + start cell (0,0)
    if bm[0, 0]:
        dx[0] += 2.0 * (X[0, 0] - Y[0, 0])  # base match at (0,0)
    for i in range(1, m):
        # sum over all j that receive diagonal edge into (i,j)
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
                X[0, i], X[0, i - 1], Y[0, j], c=c, alpha=alpha, gamma=gamma
            )
            dx[i] += occ * d_dxval  # x_val is x[i]
            dx[i - 1] += occ * d_dyprev  # y_prev is x[i-1]

    # (iii) horizontal transitions: edge (i,j-1) -> (i,j); x appears as z_other
    for i in range(m):
        for j in range(1, n):
            occ = Hocc[i, j]
            if occ == 0.0:
                continue
            # trans args: (x_val = y[j], y_prev = y[j-1], z_other = x[i])
            _, _, d_dzoth = _soft_msm_transition_cost_grads(
                Y[0, j], Y[0, j - 1], X[0, i], c=c, alpha=alpha, gamma=gamma
            )
            dx[i] += occ * d_dzoth

    return dx, R[-1, -1]
