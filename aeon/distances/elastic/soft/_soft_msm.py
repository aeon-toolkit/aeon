"""Move-split-merge (soft_msm) distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._utils import _softmin2, _softmin3
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.decorators.numba_threading import numba_thread_handler
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    gamma: float = 1.0,
) -> float:
    r"""Compute the soft-MSM distance between two time series.

    Soft-MSM is a differentiable, squared-cost relaxation of the move-split-merge
    (MSM) family. It replaces the hard minimum over alignment paths with a
    temperature-controlled soft minimum and uses squared move/split/merge costs
    (mirroring how soft-DTW relates to DTW). It is therefore **not** a relaxation
    of aeon's :func:`~aeon.distances.msm_distance` (which uses absolute costs):
    as :math:`\gamma \rightarrow 0` it converges to a *squared-cost* MSM rather
    than the standard MSM distance. Like soft-DTW it is not a proper distance -
    it can be negative and ``d(x, x) != 0`` - and it is differentiable for use in
    gradient-based methods such as soft barycentre averaging.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. ``window`` is a percentage deviation, so if ``window = 0.1``
        then 10% of the series length is the maximum warping allowed.
    c : float, default=1.0
        MSM stiffness parameter controlling the penalty for move/split/merge
        operations.
    itakura_max_slope : float or None, default=None
        Maximum slope as a proportion of the number of time points used to
        create an Itakura parallelogram on the bounding matrix.
        Must be between 0. and 1.
    gamma : float, default=1.0
        Controls the smoothness of the soft minimum over alignment paths.
        Smaller values approach the hard-min (squared-cost MSM) behaviour,
        larger values produce smoother alignments.

    Returns
    -------
    float
        soft-MSM distance between ``x`` and ``y``. May be negative for larger
        ``gamma``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import soft_msm_distance
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> y = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    >>> soft_msm_distance(x, y)
    1.3200398491544811
    """
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
    r"""Compute the soft-MSM cost matrix between two time series.

    The cost matrix contains the accumulated soft-MSM cost for all prefixes
    of ``x`` and ``y``, with the final distance given by the bottom-right
    entry.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. ``window`` is a percentage deviation, so if ``window = 0.1``
        then 10% of the series length is the maximum warping allowed.
    c : float, default=1.0
        MSM stiffness parameter controlling the penalty for move/split/merge
        operations.
    itakura_max_slope : float or None, default=None
        Maximum slope as a proportion of the number of time points used to
        create an Itakura parallelogram on the bounding matrix.
        Must be between 0. and 1.
    gamma : float, default=1.0
        Controls the smoothness of the soft minimum over alignment paths.

    Returns
    -------
    np.ndarray of shape (n_timepoints_x, n_timepoints_y)
        soft-MSM cost matrix between ``x`` and ``y``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` are not 1D or 2D arrays.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_cost_matrix(
            _x, _y, bounding_matrix=bounding_matrix, c=c, gamma=gamma
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_msm_full_cost_matrix(x, y, bounding_matrix, c, gamma)
    raise ValueError("x and y must be 1D or 2D")


@numba_thread_handler
def soft_msm_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    window: float | None = None,
    c: float = 1.0,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,
    gamma: float = 1.0,
) -> np.ndarray:
    r"""Compute the soft-MSM pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        A collection of time series instances of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)`` or a list of 1D / 2D arrays.
    y : np.ndarray or list of np.ndarray or None, default=None
        A single series or a collection of time series of shape
        ``(m_timepoints,)``, ``(m_cases, m_timepoints)`` or
        ``(m_cases, n_channels, m_timepoints)`` (or a list of such arrays).
        If None, then the soft-MSM pairwise distance between the instances
        of ``X`` is calculated.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    c : float, default=1.0
        MSM stiffness parameter controlling the penalty for move/split/merge
        operations.
    itakura_max_slope : float or None, default=None
        Maximum slope as a proportion of the number of time points used to
        create an Itakura parallelogram on the bounding matrix.
        Must be between 0. and 1.
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs
        is set to the number of CPU cores. If 1, then the function is executed
        in a single thread. If greater than 1, then the function is executed
        in parallel.
    gamma : float, default=1.0
        Controls the smoothness of the soft minimum over alignment paths.

    Returns
    -------
    np.ndarray
        soft-MSM pairwise distance matrix between the instances of ``X`` of
        shape ``(n_cases, n_cases)`` or between ``X`` and ``y`` of shape
        ``(n_cases, m_cases)``.

    Raises
    ------
    ValueError
        If ``X`` is not a 2D/3D array or list of arrays, or if ``y`` is not
        1D, 2D, 3D or a list of arrays when passed.
    """
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
    """Compute the soft-MSM alignment path between two time series.

    The alignment path is obtained by applying a standard backtracking
    procedure to the soft-MSM cost matrix.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or
        ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or
        ``(m_timepoints,)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    c : float, default=1.0
        MSM stiffness parameter controlling the penalty for move/split/merge
        operations.
    itakura_max_slope : float or None, default=None
        Maximum slope as a proportion of the number of time points used to
        create an Itakura parallelogram on the bounding matrix.
        Must be between 0. and 1.
    gamma : float, default=1.0
        Controls the smoothness of the soft minimum over alignment paths.

    Returns
    -------
    list[tuple[int, int]]
        The alignment path between the two time series where each element is a
        tuple of the index in ``x`` and the index in ``y`` that have the best
        alignment according to the soft-MSM cost matrix.
    float
        The soft-MSM distance between the two time series.
    """
    cost_matrix = soft_msm_cost_matrix(x, y, window, c, itakura_max_slope, gamma)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )


@njit(cache=True, fastmath=True)
def _soft_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
) -> float:
    cost_matrix = _soft_msm_full_cost_matrix(x, y, bounding_matrix, c, gamma)
    return cost_matrix[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _soft_msm_full_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
    gamma: float,
) -> np.ndarray:
    """Soft-MSM cost matrix for univariate ``(1, m)`` or multivariate input.

    Multivariate series use the *independent* strategy: the univariate soft-MSM
    cost matrix is summed across channels, matching
    ``msm_distance(independent=True)``.
    """
    if x.shape[0] == 1:
        return _soft_msm_cost_matrix(x, y, bounding_matrix, c, gamma)
    return _soft_multivariate_msm_cost_matrix(x, y, bounding_matrix, c, gamma)


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
    n_channels = min(x.shape[0], y.shape[0])
    for ch in range(n_channels):
        # Slice to ``(1, m)`` so the univariate kernel reads the channel.
        cost_matrix = cost_matrix + _soft_msm_cost_matrix(
            x[ch : ch + 1], y[ch : ch + 1], bounding_matrix, c, gamma
        )
    return cost_matrix


@njit(fastmath=True, cache=True)
def _soft_msm_transition_cost(x_val, y_prev, z_other, c, gamma):
    a = x_val - y_prev
    b = x_val - z_other

    # Between gate. The fixed 1e-9 floor makes this gate amplitude-dependent;
    # a scale-invariant version is proposed in
    # https://github.com/aeon-toolkit/aeon/issues/3518
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
        for j in range(n_cases):
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
def _soft_msm_transition_cost_grads(x_val, y_prev, z_other, c, gamma):
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
    Vocc = np.zeros((m, n))  # vertical edge into (i,j)
    Hocc = np.zeros((m, n))  # horizontal edge into (i,j)

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
            dx[i] += occ * d_dxval  # x_val is X[i]
            dx[i - 1] += occ * d_dyprev  # y_prev is X[i-1]

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
