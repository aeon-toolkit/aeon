r"""Weighted dynamic time warping (WDTW) distance between two time series."""

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
def wdtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.05,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the WDTW distance between two time series.

    First proposed in [1]_, weighted dynamic time warping (WDTW) uses DTW with a
    weighted pairwise distance matrix rather than a window. When
    creating the distance matrix :math:'M', a weight penalty  :math:'w_{|i-j|}' for a
    warping distance of :math:'|i-j|' is applied, so that for series
    :math:`a = <a_1, ..., a_m>` and :math:`b=<b_1,...,b_m>`,

    .. math::
        M_{i,j}=  w(|i-j|) (a_i-b_j)^2.

    A logistic weight function, proposed in [1] is used, so that a warping of :math:`x`
    places imposes a weighting of

    .. math::
        w(x)=\frac{w_{max}}{1+e^{-g(x-m/2)}},

    where :math:`w_{max}` is an upper bound on the weight (set to 1), :math:`m` is
    the series length and :math:`g` is a parameter that controls the penalty level
    for larger warpings. The greater :math:`g` is, the greater the penalty for warping.
    Once :math:`M` is found, standard dynamic time warping is applied.

    WDTW is set up so you can use it with a bounding box in addition to the weight
    function is so desired. This is for consistency with the other distance functions.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the wdtw pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g : float, default=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        WDTW distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.


    References
    ----------
    .. [1] Young-Seon Jeong, Myong K. Jeong, Olufemi A. Omitaomu, Weighted dynamic time
    warping for time series classification, Pattern Recognition, Volume 44, Issue 9,
    2011, Pages 2231-2240, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2010.09.022.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wdtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> round(wdtw_distance(x, y),1)
    356.5
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _wdtw_distance(_x, _y, bounding_matrix, g)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _wdtw_distance(x, y, bounding_matrix, g)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def wdtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.05,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the WDTW cost matrix between two time series.

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
    g : float, default=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        WDTW cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wdtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wdtw_cost_matrix(x, y)
    array([[  0.        ,   0.450166  ,   2.30044662,   6.57563393,
             14.37567559,  26.87567559,  45.32558186,  71.04956205,
            105.44507215, 149.98162593],
           [  0.450166  ,   0.        ,   0.450166  ,   2.30044662,
              6.57563393,  14.37567559,  26.87567559,  45.32558186,
             71.04956205, 105.44507215],
           [  2.30044662,   0.450166  ,   0.        ,   0.450166  ,
              2.30044662,   6.57563393,  14.37567559,  26.87567559,
             45.32558186,  71.04956205],
           [  6.57563393,   2.30044662,   0.450166  ,   0.        ,
              0.450166  ,   2.30044662,   6.57563393,  14.37567559,
             26.87567559,  45.32558186],
           [ 14.37567559,   6.57563393,   2.30044662,   0.450166  ,
              0.        ,   0.450166  ,   2.30044662,   6.57563393,
             14.37567559,  26.87567559],
           [ 26.87567559,  14.37567559,   6.57563393,   2.30044662,
              0.450166  ,   0.        ,   0.450166  ,   2.30044662,
              6.57563393,  14.37567559],
           [ 45.32558186,  26.87567559,  14.37567559,   6.57563393,
              2.30044662,   0.450166  ,   0.        ,   0.450166  ,
              2.30044662,   6.57563393],
           [ 71.04956205,  45.32558186,  26.87567559,  14.37567559,
              6.57563393,   2.30044662,   0.450166  ,   0.        ,
              0.450166  ,   2.30044662],
           [105.44507215,  71.04956205,  45.32558186,  26.87567559,
             14.37567559,   6.57563393,   2.30044662,   0.450166  ,
              0.        ,   0.450166  ],
           [149.98162593, 105.44507215,  71.04956205,  45.32558186,
             26.87567559,  14.37567559,   6.57563393,   2.30044662,
              0.450166  ,   0.        ]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _wdtw_cost_matrix(_x, _y, bounding_matrix, g)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _wdtw_cost_matrix(x, y, bounding_matrix, g)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _wdtw_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
) -> float:
    return _wdtw_cost_matrix(x, y, bounding_matrix, g)[x.shape[1] - 1, y.shape[1] - 1]


@njit(cache=True, fastmath=True)
def _wdtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    max_size = max(x_size, y_size)
    weight_vector = np.array(
        [1 / (1 + np.exp(-g * (i - max_size / 2))) for i in range(0, max_size)]
    )

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[:, i], y[:, j]
                ) * weight_vector[abs(i - j)] + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]


def wdtw_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    g: float = 0.05,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the WDTW pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the wdtw pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g : float, default=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        WDTW pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wdtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> wdtw_pairwise_distance(X)
    array([[ 0.        , 12.61266072, 51.97594869],
           [12.61266072,  0.        , 12.61266072],
           [51.97594869, 12.61266072,  0.        ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> wdtw_pairwise_distance(X, y)
    array([[144.37763524, 243.99820355, 369.60674621],
           [ 70.74504127, 144.37763524, 243.99820355],
           [ 23.10042164,  70.74504127, 144.37763524]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> wdtw_pairwise_distance(X, y_univariate)
    array([[144.37763524],
           [ 70.74504127],
           [ 23.10042164]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> wdtw_pairwise_distance(X)
    array([[  0.        ,  20.25043711, 139.70656066],
           [ 20.25043711,   0.        ,  39.64543037],
           [139.70656066,  39.64543037,   0.        ]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _wdtw_pairwise_distance(_X, window, g, itakura_max_slope, unequal_length)
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _wdtw_from_multiple_to_multiple_distance(
        _X, _y, window, g, itakura_max_slope, unequal_length
    )


@njit(cache=True, fastmath=True)
def _wdtw_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    g: float,
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
            distances[i, j] = _wdtw_distance(x1, x2, bounding_matrix, g)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _wdtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    g: float,
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
            distances[i, j] = _wdtw_distance(x1, y1, bounding_matrix, g)
    return distances


@njit(cache=True, fastmath=True)
def wdtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.05,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the WDTW alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g : float, default=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.
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
        The WDTW distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wdtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> path, dist = wdtw_alignment_path(x, y)
    >>> path
    [(0, 0), (1, 1), (2, 2), (3, 3)]
    """
    cost_matrix = wdtw_cost_matrix(x, y, window, g, itakura_max_slope)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
