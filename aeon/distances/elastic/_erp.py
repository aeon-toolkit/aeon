r"""Edit real penalty (erp) distance between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.pointwise._euclidean import _univariate_euclidean_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the ERP distance between two time series.

    Edit Distance with Real Penalty, ERP, first proposed in [1]_, attempts to align
    time series by better considering how indexes are carried forward through the
    cost matrix. Usually in the dtw cost matrix, if an alignment cannot be found the
    previous value is carried forward in  the move off the diagonal. ERP instead
    proposes the idea of gaps or sequences of points that have no matches. These
    gaps are then penalised based on their distance from the parameter :math:`g`.

    .. math::
        match  &=  D_{i-1,j-1}+ d({x_{i},y_{j}})\\
        delete &=   D_{i-1,j}+ d({x_{i},g})\\
        insert &=  D_{i,j-1}+ d({g,y_{j}})\\
        D_{i,j} &= min(match,insert, delete)

    Where :math:`D_{0,j}` and :math:`D_{i,0}` are initialised to the sum of
    distances to $g$ for each series.

    The value of :math:`g` is by default 0 in ``aeon``, but in [1]_ it is data dependent
    , selected from the range :math:`[\sigma/5, \sigma]`, where :math:`\sigma` is the
    average standard deviation of the training time series. When a
    series is multivariate (more than one channel), :math:`g` is an array where the
    :math:`j^{th}` value is the standard deviation of the :math:`j^{th}` channel.

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
        is used.
    g : float, default=0.0
        The reference constant used to penalise moves off the diagonal. The default
        is 0.
    g_arr : np.ndarray, default=None
        Array of shape ``(n_channels)``,
        Numpy array with a separate ``g`` value for each channel. Must be the
        length of the number of channels in x and y.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        ERP distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Lei Chen and Raymond Ng. 2004. On the marriage of Lp-norms and edit distance.
    In Proceedings of the Thirtieth international conference on Very large data bases
     - Volume 30 (VLDB '04). VLDB Endowment, 792–803.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[2, 2, 2, 2, 5, 6, 7, 8, 9, 10]])
    >>> erp_distance(x, y)
    4.0
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _erp_distance(_x, _y, bounding_matrix, g, g_arr)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _erp_distance(x, y, bounding_matrix, g, g_arr)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def erp_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the ERP cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window :  float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g :  float, defualt=0.0
        The reference value to penalise gaps. The default is 0.
    g_arr : np.ndarray, of shape (n_channels), default=None
        Numpy array that must be the length of the number of channels in x and y.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        ERP cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> erp_cost_matrix(x, y)
    array([[ 0.,  2.,  5.,  9., 14., 20., 27., 35., 44., 54.],
           [ 2.,  0.,  3.,  7., 12., 18., 25., 33., 42., 52.],
           [ 5.,  3.,  0.,  4.,  9., 15., 22., 30., 39., 49.],
           [ 9.,  7.,  4.,  0.,  5., 11., 18., 26., 35., 45.],
           [14., 12.,  9.,  5.,  0.,  6., 13., 21., 30., 40.],
           [20., 18., 15., 11.,  6.,  0.,  7., 15., 24., 34.],
           [27., 25., 22., 18., 13.,  7.,  0.,  8., 17., 27.],
           [35., 33., 30., 26., 21., 15.,  8.,  0.,  9., 19.],
           [44., 42., 39., 35., 30., 24., 17.,  9.,  0., 10.],
           [54., 52., 49., 45., 40., 34., 27., 19., 10.,  0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _erp_cost_matrix(_x, _y, bounding_matrix, g, g_arr)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _erp_cost_matrix(x, y, bounding_matrix, g, g_arr)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    g: float,
    g_arr: Optional[np.ndarray],
) -> float:
    return _erp_cost_matrix(x, y, bounding_matrix, g, g_arr)[
        x.shape[1] - 1, y.shape[1] - 1
    ]


@njit(cache=True, fastmath=True)
def _erp_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    g: float,
    g_arr: Optional[np.ndarray],
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    gx_distance, x_sum = _precompute_g(x, g, g_arr)
    gy_distance, y_sum = _precompute_g(y, g, g_arr)

    cost_matrix[1:, 0] = x_sum
    cost_matrix[0, 1:] = y_sum
    cost_matrix[0, 0] = 0.0

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1]
                    + _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]),
                    cost_matrix[i - 1, j] + gx_distance[i - 1],
                    cost_matrix[i, j - 1] + gy_distance[j - 1],
                )

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _precompute_g(
    x: np.ndarray, g: float, g_array: Optional[np.ndarray]
) -> tuple[np.ndarray, float]:
    gx_distance = np.zeros(x.shape[1])
    if g_array is None:
        g_arr = np.full(x.shape[0], g)
    else:
        if g_array.shape[0] != x.shape[0]:
            raise ValueError("g must be a float or an array with shape (x.shape[0],)")
        g_arr = g_array
    x_sum = 0

    for i in range(x.shape[1]):
        temp = _univariate_euclidean_distance(x[:, i], g_arr)
        gx_distance[i] = temp
        x_sum += temp
    return gx_distance, x_sum


def erp_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the ERP pairwise distance between a set of time series.

    The optimal value of g is selected from the range [σ/5, σ], where σ is the
    standard deviation of the training data. When there is > 1 channel, g should
    be a np.ndarray where the nth value is the standard deviation of the nth
    channel.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the erp pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g : float, default=0.0.
        The reference value to penalise gaps. The default is 0.
    g_arr : np.ndarray, of shape (n_channels), default=None
        Numpy array that must be the length of the number of channels in x and y.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        ERP pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> erp_pairwise_distance(X)
    array([[ 0.,  9., 18.],
           [ 9.,  0.,  9.],
           [18.,  9.,  0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> erp_pairwise_distance(X, y)
    array([[30., 39., 48.],
           [21., 30., 39.],
           [12., 21., 30.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> erp_pairwise_distance(X, y_univariate)
    array([[30.],
           [21.],
           [12.]])
    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> erp_pairwise_distance(X)
    array([[ 0., 16., 44.],
           [16.,  0., 28.],
           [44., 28.,  0.]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )
    if y is None:
        return _erp_pairwise_distance(
            _X, window, g, g_arr, itakura_max_slope, unequal_length
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _erp_from_multiple_to_multiple_distance(
        _X, _y, window, g, g_arr, itakura_max_slope, unequal_length
    )


@njit(cache=True, fastmath=True)
def _erp_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    g: float,
    g_arr: Optional[np.ndarray],
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
            distances[i, j] = _erp_distance(x1, x2, bounding_matrix, g, g_arr)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _erp_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    g: float,
    g_arr: Optional[np.ndarray],
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
            distances[i, j] = _erp_distance(x1, y1, bounding_matrix, g, g_arr)
    return distances


@njit(cache=True, fastmath=True)
def erp_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.0,
    g_arr: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the ERP alignment path between two time series.

    The optimal value of g is selected from the range [σ/5, σ], where σ is the
    The optimal value of g is selected from the range [σ/5, σ], where σ is the
    standard deviation of the training data. When there is > 1 channel, g should
    be a np.ndarray where the nth value is the standard deviation of the nth
    channel.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g : float, default=0.0.
        The reference value to penalise gaps. The default is 0.
    g_arr : np.ndarray, of shape (n_channels), default=None
        Numpy array that must be the length of the number of channels in x and y.
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
        The erp distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> erp_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 2.0)
    """
    cost_matrix = erp_cost_matrix(x, y, window, g, g_arr)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
