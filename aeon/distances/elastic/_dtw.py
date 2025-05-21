r"""Dynamic time warping (DTW) between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.pointwise._squared import _univariate_squared_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the DTW distance between two time series.

    DTW is the most widely researched and used elastic distance method. It mitigates
    distortions in the time axis by realligning (warping) the series to best match
    each other. A good background into DTW can be found in [1]_. For two series,
    possibly of unequal length,
    :math:`\mathbf{x}=\{x_1,x_2,\ldots,x_n\}` and
    :math:`\mathbf{y}=\{y_1,y_2, \ldots,y_m\}` DTW first calculates
    :math:`M(\mathbf{x},\mathbf{y})`, the :math:`n \times m`
    pointwise distance matrix between series :math:`\mathbf{x}` and :math:`\mathbf{y}`,
    where :math:`M_{i,j}=   (x_i-y_j)^2`.

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

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_distance
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> dtw_distance(x, y) # 1D series
    768.0
    >>> x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 1, 0, 2, 0]])
    >>> y = np.array([[11, 12, 13, 14],[7, 8, 9, 20],[1, 3, 4, 5]] )
    >>> dtw_distance(x, y) # 2D series with 3 channels, unequal length
    564.0
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _dtw_distance(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _dtw_distance(x, y, bounding_matrix)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    r"""Compute the DTW cost matrix between two time series.

    The cost matrix is the pairwise Euclidean distance between all points
    :math:`M_{i,j}=(x_i-x_j)^2`. It is used in the DTW path calculations.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
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
        dtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> dtw_cost_matrix(x, y)
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
        return _dtw_cost_matrix(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(x, y, bounding_matrix)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _dtw_distance(x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray) -> float:
    return _dtw_cost_matrix(x, y, bounding_matrix)[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _dtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[:, i], y[:, j]
                ) + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]


def dtw_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    r"""Compute the DTW pairwise distance between a set of time series.

    By default, this takes a collection of :math:`n` time series :math:`X` and returns a
    matrix
    :math:`D` where :math:`D_{i,j}` is the DTW distance between the :math:`i^{th}`
    and the :math:`j^{th}` series in :math:`X`. If :math:`X` is 2 dimensional,
    it is assumed to be a collection of univariate series with shape ``(n_cases,
    n_timepoints)``. If it is 3 dimensional, it is assumed to be shape ``(n_cases,
    n_channels, n_timepoints)``.

    This function has an optional argument, :math:`y`, to allow calculation of the
    distance matrix between :math:`X` and one or more series stored in :math:`y`. If
    :math:`y` is 1 dimensional, we assume it is a single univariate series and the
    distance matrix returned is shape ``(n_cases,1)``. If it is 2D, we assume it
    is a collection of univariate series with shape ``(m_cases, m_timepoints)``
    and the distance ``(n_cases,m_cases)``. If it is 3 dimensional,
    it is assumed to be shape ``(m_cases, m_channels, m_timepoints)``.

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
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray
        DTW pairwise matrix between the instances of X of shape
        ``(n_cases, n_cases)`` or between X and y of shape ``(n_cases,
        n_cases)``.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array and if y is not 1D, 2D or 3D arrays when passing y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> dtw_pairwise_distance(X)
    array([[  0.,  26., 108.],
           [ 26.,   0.,  26.],
           [108.,  26.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> dtw_pairwise_distance(X, y)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> dtw_pairwise_distance(X, y_univariate)
    array([[300.],
           [147.],
           [ 48.]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> dtw_pairwise_distance(X)
    array([[  0.,  42., 292.],
           [ 42.,   0.,  83.],
           [292.,  83.,   0.]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _dtw_pairwise_distance(_X, window, itakura_max_slope, unequal_length)
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _dtw_from_multiple_to_multiple_distance(
        _X, _y, window, itakura_max_slope, unequal_length
    )


@njit(cache=True, fastmath=True)
def _dtw_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _dtw_distance(x1, x2, bounding_matrix)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _dtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _dtw_distance(x1, y1, bounding_matrix)
    return distances


@njit(cache=True, fastmath=True)
def dtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the DTW alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
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
        The DTW distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> dtw_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    cost_matrix = dtw_cost_matrix(x, y, window, itakura_max_slope)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
