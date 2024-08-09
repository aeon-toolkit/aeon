"""Parameterised distances for Proximity Forest 2.0."""

from typing import Optional

import numpy as np
from numba import njit

from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._minkowski import _univariate_minkowski_distance


@njit(cache=True, fastmath=True)
def _dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    p: Optional[float] = 2.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the DTW distance between two time series.

    DTW is the most widely researched and used elastic distance measure. It mitigates
    distortions in the time axis by realligning (warping) the series to best match
    each other. A good background into DTW can be found in [1]_. For two series,
    possibly of unequal length,
    :math:`\mathbf{x}=\{x_1,x_2,\ldots,x_n\}` and
    :math:`\mathbf{y}=\{y_1,y_2, \ldots,y_m\}` DTW first calculates
    :math:`M(\mathbf{x},\mathbf{y})`, the :math:`n \times m`
    pointwise distance matrix between series :math:`\mathbf{x}` and :math:`\mathbf{y}`,
    where :math:`M_{i,j}=   (x_i-y_j)^p`.

    A warping path

    .. math::
        P = <(e_1, f_1), (e_2, f_2), \ldots, (e_s, f_s)>

    is a set of pairs of indices that  define a traversal of matrix :math:`M`. A
    valid warping path must start at location :math:`(1,1)` and end at point :math:`(
    n,m)` and not backtrack, i.e. :math:`0 \leq e_{i+1}-e_{i} \leq 1` and :math:`0
    \leq f_{i+1}- f_i \leq 1` for all :math:`1< i < m`.

    The DTW distance between series is the path through :math:`M` that minimizes the
    total distance. The distance for any path :math:`P` of length :math:`s` is

    .. math::
        D_P(\mathbf{x},\mathbf{y}, M) =\sum_{i=1}^s M_{e_i,f_i}

    If :math:`\mathcal{P}` is the space of all possible paths, the DTW path :math:`P^*`
    is the path that has the minimum distance, hence the DTW distance between series is

    .. math::
        d_{dtw}(\mathbf{x}, \mathbf{x}) =D_{P*}(\mathbf{x},\mathbf{x}, M).

    The optimal warping path :math:`P^*` can be found exactly through a dynamic
    programming formulation. This can be a time consuming operation, and it is common to
    put a restriction on the amount of warping allowed. This is implemented through
    the bounding_matrix structure, that supplies a mask for allowable warpings.
    The most common bounding strategies include the Sakoe-Chiba band [2]_. The width
    of the allowed warping is controlled through the ``window`` parameter
    which sets the maximum proportion of warping allowed.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    p : float, default=2.0
        The order of the norm of the difference
        (default is 2.0, which represents the Euclidean distance).
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
        DTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Ratanamahatana C and Keogh E.: Three myths about dynamic time warping data
    mining, Proceedings of 5th SIAM International Conference on Data Mining, 2005.

    .. [2] Sakoe H. and Chiba S.: Dynamic programming algorithm optimization for
    spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal
    Processing 26(1):43–49, 1978.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(x, y, p, bounding_matrix)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(x, y, p, bounding_matrix)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _dtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, p: float, bounding_matrix: np.ndarray
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    _w = np.ones_like(x)
    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_minkowski_distance(
                    x[:, i], y[:, j], p, _w[:, i]
                ) + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _adtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    p: Optional[float] = 2.0,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> float:
    r"""Compute the ADTW distance between two time series.

    Amercing Dynamic Time Warping (ADTW) [1]_ is a variant of DTW that uses a
    explicit warping penalty to encourage or discourage warping. The warping
    penalty is a constant value that is added to the cost of warping. A high
    value will encourage the algorithm to warp less and if the value is low warping
    is more likely.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    p : float, default=2.0
        The order of the norm of the difference
        (default is 2.0, which represents the Euclidean distance).
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0.0 and 1.0
    warp_penalty: float, default=1.0
        Penalty for warping. A high value will mean less warping.

    Returns
    -------
    float
        ADTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Matthieu Herrmann, Geoffrey I. Webb: Amercing: An intuitive and effective
    constraint for dynamic time warping, Pattern Recognition, Volume 137, 2023.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _adtw_cost_matrix(x, y, p, bounding_matrix, warp_penalty)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _adtw_cost_matrix(x, y, p, bounding_matrix, warp_penalty)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _adtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    p: float,
    bounding_matrix: np.ndarray,
    warp_penalty: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    _w = np.ones_like(x)
    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_minkowski_distance(
                    x[:, i], y[:, j], p, _w[:, i]
                ) + min(
                    cost_matrix[i, j + 1] + warp_penalty,
                    cost_matrix[i + 1, j] + warp_penalty,
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]
