r"""Dynamic time warping with Global Invariances (DTW-GI) between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._dtw import dtw_alignment_path, dtw_cost_matrix
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def _path2mat(
    path: list[tuple[int, int]],
    x_timepoints: int,
    y_timepoints: int,
):
    r"""Convert a warping alignment path to a binary warping matrix."""
    w = np.zeros((x_timepoints, y_timepoints))
    for i, j in path:
        w[i, j] = 1
    return w


@njit(cache=True, fastmath=True)
def _dtw_gi(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    init_p: np.ndarray = None,
    max_iter: int = 20,
    use_bias: bool = False,
):
    r"""
    Compute Dynamic Time Warping with Global Invariance between the two time series.

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
    init_p : array-like of shape (x_channels, y_channels), default=None
        Initial linear transformation. If None, the identity matrix is used.
    max_iter : int, default=20
        Maximum number of iterations for the iterative optimization.
    use_bias : bool, default=False
        If True, the feature space map is affine (with a bias term).

    Returns
    -------
    - w_pi: binary warping matrix of shape (n0, n1)
    - p: the final linear (Stiefel) matrix of shape (x_channels, y_channels)
    - cost: final DTW cost considering global invariances

    If use_bias is True, also returns:
      - bias

    """
    if x.ndim == 1 and y.ndim == 1:
        x_ = x.reshape((1, x.shape[0]))
        y_ = y.reshape((1, y.shape[0]))
    elif x.ndim == 2 and y.ndim == 2:
        x_ = x
        y_ = y
    else:
        raise ValueError("x and y must be 1D or 2D")

    x_channels, x_timepoints = x_.shape
    y_channels, y_timepoints = y_.shape

    x_m = np.sum(x_, axis=1) / x_.shape[1]
    x_m = x_m.reshape((-1, 1))
    y_m = np.sum(y_, axis=1) / y_.shape[1]
    y_m = y_m.reshape((-1, 1))

    w_pi = np.zeros((x_timepoints, y_timepoints))
    if init_p is None:
        p = np.eye(x_channels, y_channels, dtype=np.float64)
    else:
        p = init_p

    y_ = y_.astype(np.float64)
    x_ = x_.astype(np.float64)

    bias = np.zeros((x_channels, 1))

    for _ in range(max_iter):
        w_pi_old = w_pi.copy()
        y_transformed = p.dot(y_) + bias

        path, cost = dtw_alignment_path(x_, y_transformed, window, itakura_max_slope)
        w_pi = _path2mat(path, x_timepoints, y_timepoints)

        if np.allclose(w_pi, w_pi_old):
            break

        if use_bias:
            m = (x_ - x_m).dot(w_pi).dot((y_ - y_m).T)
        else:
            m = x_.dot(w_pi).dot(y_.T)

        u, sigma, vt = np.linalg.svd(m, full_matrices=False)
        p = u.dot(vt)
        if use_bias:
            bias = x_m - p.dot(y_m)

    y_trans = p.dot(y_) + bias
    path, cost = dtw_alignment_path(x_, y_trans, window, itakura_max_slope)

    if use_bias:
        return w_pi, p, bias, cost, x_, y_trans
    else:
        dummy_bias = np.zeros((x_channels, 1), dtype=np.float64)
        return w_pi, p, dummy_bias, cost, x_, y_trans


@njit(cache=True, fastmath=True)
def dtw_gi_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    init_p: np.ndarray = None,
    max_iter: int = 20,
    use_bias: bool = False,
) -> float:
    r"""Compute the DTW_GI distance between two time series x and y.

    The DTW_gi distance between 2 timeseries x and y is the similarity
    measure that estimates both temporal alignment and does feature space
    transformation between time series simultaneously. This means that the
    time series do not have to lie in the same ambient space.
    A good background into DTW with global invariances can be found in [1]_.
    This implementation is inspired by [2]_.

    For example, if we have two time series x and y of lengths n and m
    respectively, and we assume that the time series do not lie in the
    same ambient space. Lets assume that features of x lie in :math:`\mathbb{R}^p`
    and features of y lie in :math:`\mathbb{R}^q`. To compare the two time series,
    we need to find an optimum mapping from the feature space of y to the feature space
    where features of x lie. So think of it as optimizing on a family of functions F
    that map features from y onto the feature space in which features of x
    lie. (This is just one way to do it, the mapping can be
    in the opposite direction as well. But this function assumes the former way).

    More formally, we define Dynamic Time Warping with Global Invariances (DTW-GI)
    as the solution of the following joint optimization problem:

    :math:`\text{DTW-GI}(\mathbf{x}, \mathbf{y}) =
    \min_{f \in \mathcal{F}, \pi \in \mathcal{A}(\mathbf{x}, \mathbf{y})}
    \sqrt{\sum_{(i,j) \in \pi} d(x_i, f(y_j))^2},`

    This similarity measure estimates temporal alignment
    with feature space transformation between time series
    allowing the alignment of time series that do not exist in the
    same ambient space.

    Note: The optimal warping path :math:`P^*` for a given cost matrix can be found
    exactly through a dynamic programming formulation. This can be a time consuming
    operation, and it is common to put a restriction on the amount of warping allowed.
    This is implemented through the bounding_matrix structure, that supplies a mask for
    allowable warpings. The most common bounding strategies include the
    Sakoe-Chiba band [3]_. The width of the allowed warping is controlled through the
     ``window`` parameter which sets the maximum proportion of warping allowed.

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
    init_p : array-like of shape (x_channels, y_channels), default=None
        Initial linear transformation. If None, the identity matrix is used.
    max_iter : int, default=20
        Maximum number of iterations for the iterative optimization.
    use_bias : bool, default=False
        If True, the feature space map is affine (with a bias term).

    Returns
    -------
    float
        DTW_GI distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] T. Vayer, R. Tavenard, L. Chapel, N. Courty, R. Flamary, and Y. Soullard,
    “Time Series Alignment with Global Invariances,” arXiv.org, 2020.
    https://arxiv.org/abs/2002.03848

    .. [2] Romain Tavenard, “DTW with Global Invariances,” Github.io, Dec. 17, 2020.
    https://rtavenar.github.io/hdr/parts/01/dtw/dtw_gi.html

    .. [3] Sakoe H. and Chiba S.: Dynamic programming algorithm optimization for
    spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal
    Processing 26(1):43-49, 1978.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_gi_distance
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> dtw_gi_distance(x, y) # 1D series
    768.0
    >>> x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 1, 0, 2, 0]])
    >>> y = np.array([[11, 12, 13, 14],[7, 8, 9, 20],[1, 3, 4, 5]] )
    >>> round(dtw_gi_distance(x, y), 1) # 2D series with 3 channels, unequal length
    359.2
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        return _dtw_gi(_x, _y, window, itakura_max_slope, init_p, max_iter, use_bias)[3]
    if x.ndim == 2 and y.ndim == 2:
        return _dtw_gi(x, y, window, itakura_max_slope, init_p, max_iter, use_bias)[3]
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def dtw_gi_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    init_p: np.ndarray = None,
    max_iter: int = 20,
    use_bias: bool = False,
) -> np.ndarray:
    r"""Compute the DTW_GI cost matrix between two time series.

    The cost matrix is the pairwise Euclidean distance between all points
    :math:`M_{i,j}=(x_i-y_{\text{trans},j})^2`. Where `y_trans` is the time
    series we get by finding the optimal mapping from feature space of y
    to feature space where features of x lie. It is used in the DTW GI
    path calculations.

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
    init_p : array-like of shape (x_channels, y_channels), default=None
        Initial linear transformation. If None, the identity matrix is used.
    max_iter : int, default=20
        Maximum number of iterations for the iterative optimization.
    use_bias : bool, default=False
        If True, the feature space map is affine (with a bias term).

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        dtw gi cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_gi_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> dtw_gi_cost_matrix(x, y)
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
    _, _, _, _, xnew, y_trans = _dtw_gi(
        x, y, window, itakura_max_slope, init_p, max_iter, use_bias
    )

    return dtw_cost_matrix(xnew, y_trans, window, itakura_max_slope)


def dtw_gi_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    unequal_length: bool = None,
    init_p: np.ndarray = None,
    max_iter: int = 20,
    use_bias: bool = False,
) -> np.ndarray:
    r"""Compute the DTW_GI pairwise distance between a set of time series.

    By default, this takes a collection of :math:`n` time series :math:`X` and returns a
    matrix
    :math:`D` where :math:`D_{i,j}` is the DTW_GI distance between the :math:`i^{th}`
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
        If None, then the dtw gi pairwise distance between the instances of X is
        calculated.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    init_p : array-like of shape (x_channels, y_channels), default=None
        Initial linear transformation. If None, the identity matrix is used.
    max_iter : int, default=20
        Maximum number of iterations for the iterative optimization.
    use_bias : bool, default=False
        If True, the feature space map is affine (with a bias term).

    Returns
    -------
    np.ndarray
        DTW_GI pairwise matrix between the instances of X of shape
        ``(n_cases, n_cases)`` or between X and y of shape ``(n_cases,
        n_cases)``.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array and if y is not 1D, 2D or 3D arrays when passing y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_gi_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> dtw_gi_pairwise_distance(X)
    array([[  0.,  26., 108.],
           [ 26.,   0.,  26.],
           [108.,  26.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> dtw_gi_pairwise_distance(X, y)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> dtw_gi_pairwise_distance(X, y_univariate)
    array([[300.],
           [147.],
           [ 48.]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> dtw_gi_pairwise_distance(X)
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
        return _dtw_gi_pairwise_distance(
            _X, window, itakura_max_slope, unequal_length, init_p, max_iter, use_bias
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _dtw_gi_from_multiple_to_multiple_distance(
        _X, _y, window, itakura_max_slope, unequal_length, init_p, max_iter, use_bias
    )


@njit(cache=True, fastmath=True)
def _dtw_gi_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    unequal_length: bool = None,
    init_p: np.ndarray = None,
    max_iter: int = 20,
    use_bias: bool = False,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            distances[i, j] = dtw_gi_distance(
                x1, y1, window, itakura_max_slope, init_p, max_iter, use_bias
            )
    return distances


@njit(cache=True, fastmath=True)
def _dtw_gi_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    unequal_length: bool = None,
    init_p: np.ndarray = None,
    max_iter: int = 20,
    use_bias: bool = False,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            distances[i, j] = dtw_gi_distance(
                x1, x2, window, itakura_max_slope, init_p, max_iter, use_bias
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def dtw_gi_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    init_p: np.ndarray = None,
    max_iter: int = 20,
    use_bias: bool = False,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the DTW_GI alignment path between two time series.

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
    init_p : array-like of shape (x_channels, y_channels), default=None
        Initial linear transformation. If None, the identity matrix is used.
    max_iter : int, default=20
        Maximum number of iterations for the iterative optimization.
    use_bias : bool, default=False
        If True, the feature space map is affine (with a bias term).

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The DTW_GI distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import dtw_gi_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> dtw_gi_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    w_pi, _, _, cost, _, _ = _dtw_gi(
        x, y, window, itakura_max_slope, init_p, max_iter, use_bias
    )
    min_alignment_path = []
    for i in range(len(w_pi)):
        for j in range(len(w_pi[0])):
            if w_pi[i, j] == 1:
                min_alignment_path.append((i, j))

    return min_alignment_path, cost
