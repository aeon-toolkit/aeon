r"""Soft dynamic time warping (soft-DTW) between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic.soft._utils import _softmin3
from aeon.distances.pointwise._squared import _univariate_squared_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.numba._threading import threaded
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def soft_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> float:
    r"""Compute the soft-DTW distance between two time series.

    Soft-DTW [1]_ is a soft version of the DTW distance. Soft-DTW uses a soft min
    instead of a hard min found in DTW. This makes soft-DTW differentiable. Formally
    soft-DTW is defined as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    When :math:`\gamma = 0`, :math:`\min^\gamma` this is equivalent to the
    hard-min operator and therefore the DTW distance is returned.


    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    gamma : float, default=1.0
        Controls the smoothness of the warping. A value of 0.0 is equivalent to DTW.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        soft-DTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import soft_dtw_distance
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> soft_dtw_distance(x, y)
    767.43894416832
    >>> x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 1, 0, 2, 0]])
    >>> y = np.array([[11, 12, 13, 14],[7, 8, 9, 20],[1, 3, 4, 5]] )
    >>> soft_dtw_distance(x, y)
    563.9999999999623
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_dtw_distance(_x, _y, bounding_matrix, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_dtw_distance(x, y, bounding_matrix, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def soft_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> np.ndarray:
    r"""Compute the soft-DTW cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    gamma : float, default=1.0
        Controls the smoothness of the warping. A value of 0.0 is equivalent to DTW.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1``,
        10% of the series length is the max warping allowed.
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        soft-DTW cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import soft_dtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> soft_dtw_cost_matrix(x, y)
    array([[  0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204., 285.],
           [  1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140., 204.],
           [  5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91., 140.],
           [ 14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.,  91.],
           [ 30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.,  55.],
           [ 55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.,  30.],
           [ 91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.,  14.],
           [140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.,   5.],
           [204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.,   1.],
           [285., 204., 140.,  91.,  55.,  30.,  14.,   5.,   1.,   0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_dtw_cost_matrix(_x, _y, bounding_matrix, gamma)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _soft_dtw_cost_matrix(x, y, bounding_matrix, gamma)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _soft_dtw_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, gamma: float
) -> float:
    return _soft_dtw_cost_matrix(x, y, bounding_matrix, gamma)[
        x.shape[1] - 1, y.shape[1] - 1
    ]


@njit(cache=True, fastmath=True)
def _soft_dtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, gamma: float
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                cost_matrix[i, j] = _univariate_squared_distance(
                    x[:, i - 1], y[:, j - 1]
                ) + _softmin3(
                    cost_matrix[i - 1, j],
                    cost_matrix[i - 1, j - 1],
                    cost_matrix[i, j - 1],
                    gamma,
                )
    return cost_matrix[1:, 1:]


@threaded
def soft_dtw_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    r"""Compute the soft-DTW pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the dtw pairwise distance between the instances of X is
        calculated.
    gamma : float, default=1.0
        Controls the smoothness of the warping. A value of 0.0 is equivalent to DTW.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. If 1, then the function is executed in a single
        thread. If greater than 1, then the function is executed in parallel.

    Returns
    -------
    np.ndarray
        soft-DTW pairwise matrix between the instances of X of shape
        ``(n_cases, n_cases)`` or between X and y of shape ``(n_cases,
        n_cases)``.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array and if y is not 1D, 2D or 3D arrays when passing y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import soft_dtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> soft_dtw_pairwise_distance(X)
    array([[  0.        ,  25.44075098, 107.99999917],
           [ 25.44075098,   0.        ,  25.44075098],
           [107.99999917,  25.44075098,   0.        ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> soft_dtw_pairwise_distance(X, y)
    array([[300.        , 507.        , 768.        ],
           [147.        , 300.        , 507.        ],
           [ 47.87067418, 147.        , 300.        ]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> soft_dtw_pairwise_distance(X, y_univariate)
    array([[300.        ],
           [147.        ],
           [ 47.87067418]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> soft_dtw_pairwise_distance(X)
    array([[  0.        ,  41.44055555, 291.99999969],
           [ 41.44055555,   0.        ,  82.43894439],
           [291.99999969,  82.43894439,   0.        ]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _soft_dtw_pairwise_distance(
            _X, window, itakura_max_slope, unequal_length, gamma
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _soft_dtw_from_multiple_to_multiple_distance(
        _X, _y, window, itakura_max_slope, unequal_length, gamma
    )


@njit(cache=True, fastmath=True, parallel=True)
def _soft_dtw_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: float | None,
    itakura_max_slope: float | None,
    unequal_length: bool,
    gamma: float,
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
            distances[i, j] = _soft_dtw_distance(x1, x2, bounding_matrix, gamma)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _soft_dtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: float | None,
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
            distances[i, j] = _soft_dtw_distance(x1, y1, bounding_matrix, gamma)
    return distances


@njit(cache=True, fastmath=True)
def _soft_dtw_cost_matrix_return_dist_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    dist_matrix = np.zeros((x_size, y_size))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                dist = _univariate_squared_distance(x[:, i - 1], y[:, j - 1])
                dist_matrix[i - 1, j - 1] = dist
                cost_matrix[i, j] = dist + _softmin3(
                    cost_matrix[i - 1, j],
                    cost_matrix[i - 1, j - 1],
                    cost_matrix[i, j - 1],
                    gamma,
                )
    return cost_matrix[1:, 1:], dist_matrix


@njit(cache=True, fastmath=True)
def soft_dtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the soft-DTW alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    gamma : float, default=1.0
        Controls the smoothness of the warping. A value of 0.0 is equivalent to DTW.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The soft-DTW distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import soft_dtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> soft_dtw_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 2.433544698954205)
    """
    cost_matrix = soft_dtw_cost_matrix(x, y, gamma, window, itakura_max_slope)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )


# --- fix a small bug: use the reshaped _x/_y when computing matrices ---
@njit(cache=True, fastmath=True)
def soft_dtw_alignment_matrix(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> tuple[np.ndarray, float]:

    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        cost_matrix, dist_matrix = _soft_dtw_cost_matrix_return_dist_matrix(
            _x, _y, bounding_matrix, gamma  # <- was (x, y)
        )
        return (
            _soft_gradient(dist_matrix, cost_matrix, gamma),
            cost_matrix[_x.shape[-1] - 1, _y.shape[-1] - 1],
        )

    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        cost_matrix, dist_matrix = _soft_dtw_cost_matrix_return_dist_matrix(
            x, y, bounding_matrix, gamma
        )
        return (
            _soft_gradient(dist_matrix, cost_matrix, gamma),
            cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
        )

    return (np.zeros((0, 0)), 0.0)


@njit(cache=True, fastmath=True)
def _soft_gradient(
    distance_matrix: np.ndarray, cost_matrix: np.ndarray, gamma: float
) -> np.ndarray:
    m, n = distance_matrix.shape
    E = np.zeros((m, n), dtype=float)

    E[m - 1, n - 1] = 1.0

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            r_ij = cost_matrix[i, j]
            E_ij = E[i, j]

            if i + 1 < m:
                w_horizontal = np.exp(
                    (cost_matrix[i + 1, j] - r_ij - distance_matrix[i + 1, j]) / gamma
                )
                E_ij += E[i + 1, j] * w_horizontal

            if j + 1 < n:
                w_vertical = np.exp(
                    (cost_matrix[i, j + 1] - r_ij - distance_matrix[i, j + 1]) / gamma
                )
                E_ij += E[i, j + 1] * w_vertical

            if (i + 1 < m) and (j + 1 < n):
                w_diag = np.exp(
                    (cost_matrix[i + 1, j + 1] - r_ij - distance_matrix[i + 1, j + 1])
                    / gamma
                )
                E_ij += E[i + 1, j + 1] * w_diag

            E[i, j] = E_ij

    return E


def soft_dtw_grad_x(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """
    Gradient (Jacobian) of soft-DTW distance w.r.t. the first series x.

    Returns (dx, distance); dx has shape (len(x),) for univariate or (C, T) for
    multivariate.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    gamma : float, default=1.0
        Controls the smoothness of the warping. A value of 0.0 is equivalent to DTW.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    dx : np.ndarray
        The gradient of the soft-DTW distance with respect to ``x``.
        If ``x`` is univariate (``x.ndim == 1``), the shape is ``(n_timepoints,)``.
        If ``x`` is multivariate (``x.ndim == 2``), the shape is
        ``(n_channels, n_timepoints)``.
    distance: float
        The soft-DTW distance between the two time series.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for a differentiable soft minimum.")
    if x.ndim == 1 and y.ndim == 1:
        X = x.reshape((1, x.shape[0]))
        Y = y.reshape((1, y.shape[0]))
    else:
        X = x
        Y = y
    dx, s_xy = _soft_dtw_grad_x(X, Y, gamma, window, itakura_max_slope)
    return (dx.ravel(), s_xy) if x.ndim == 1 else (dx, s_xy)


@njit(cache=True, fastmath=True)
def _soft_dtw_grad_x(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    # bounding + forward DP
    bm = create_bounding_matrix(X.shape[1], Y.shape[1], window, itakura_max_slope)
    cost_matrix, dist_matrix = _soft_dtw_cost_matrix_return_dist_matrix(X, Y, bm, gamma)
    s_xy = cost_matrix[X.shape[1] - 1, Y.shape[1] - 1]

    # backward expected-alignment (node occupancy)
    E = _soft_gradient(dist_matrix, cost_matrix, gamma)  # shape (T, U)

    C, T = X.shape[0], X.shape[1]
    U = Y.shape[1]
    dx = np.zeros_like(X)

    # ∂s/∂X[:, i] = 2 * sum_j (X[:, i] - Y[:, j]) * E[i, j]
    for i in range(T):
        acc = np.zeros(C)
        for j in range(U):
            acc += (X[:, i] - Y[:, j]) * E[i, j]
        dx[:, i] = 2.0 * acc

    return dx, s_xy
