"""Derivative Dynamic Time Warping (DDTW) distance."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._dtw import (
    _dtw_cost_matrix,
    _dtw_distance,
    create_bounding_matrix,
)
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def ddtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the DDTW distance between two time series.

    Derivative dynamic time warping (DDTW) is an adaptation of DTW originally proposed
    in [1]_. DDTW takes a version of the first derivatives of the series
    prior to performing standard DTW. The derivative function, defined in [1]_,
    is:

    .. math::
        d_{i}(x) = \frac{{}(x_{i} - x_{i-1} + (x_{i+1} - x_{i-1})/2)}{2}

    where :math:`x` is the original time series and :math:`d_x` is the derived time
    series.

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
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        ddtw distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If n_timepoints or m_timepoints are less than 2.

    References
    ----------
    .. [1] Keogh, Eamonn & Pazzani, Michael. (2002). Derivative Dynamic Time Warping.
        First SIAM International Conference on Data Mining.
        1. 10.1137/1.9781611972719.1.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import ddtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[42, 23, 21, 55, 1, 19, 33, 34, 29, 19]])
    >>> round(ddtw_distance(x, y))
    2180
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = average_of_slope(x.reshape((1, x.shape[0])))
        _y = average_of_slope(y.reshape((1, y.shape[0])))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _dtw_distance(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        _x = average_of_slope(x)
        _y = average_of_slope(y)
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _dtw_distance(_x, _y, bounding_matrix)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def ddtw_cost_matrix(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    r"""Compute the DDTW cost matrix between two time series.

    This involves taking the difference of the series then using the same cost
    function as DTW.

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
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        ddtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If n_timepoints or m_timepoints are less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import ddtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> ddtw_cost_matrix(x, y)
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = average_of_slope(x.reshape((1, x.shape[0])))
        _y = average_of_slope(y.reshape((1, y.shape[0])))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        _x = average_of_slope(x)
        _y = average_of_slope(y)
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(_x, _y, bounding_matrix)
    raise ValueError("x and y must be 1D or 2D")


def ddtw_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the DDTW pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the ddtw pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        ddtw pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.
        If n_timepoints is less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import ddtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[49, 58, 61]], [[73, 82, 99]]])
    >>> ddtw_pairwise_distance(X)
    array([[  0.  ,  42.25, 100.  ],
           [ 42.25,   0.  ,  12.25],
           [100.  ,  12.25,   0.  ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[19, 12, 39]],[[40, 51, 69]], [[79, 28, 91]]])
    >>> y = np.array([[[110, 15, 123]],[[14, 150, 116]], [[9917, 118, 29]]])
    >>> ddtw_pairwise_distance(X, y)
    array([[2.09306250e+03, 8.46400000e+03, 5.43611290e+07],
           [3.24900000e+03, 6.52056250e+03, 5.45271481e+07],
           [4.73062500e+02, 1.34560000e+04, 5.40078010e+07]])

    >>> X = np.array([[[10, 22, 399]],[[41, 500, 1316]], [[117, 18, 9]]])
    >>> y_univariate = np.array([100, 11, 199])
    >>> ddtw_pairwise_distance(X, y_univariate)
    array([[ 15129.    ],
           [322624.    ],
           [  3220.5625]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> ddtw_pairwise_distance(X)
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _ddtw_pairwise_distance(_X, window, itakura_max_slope, unequal_length)

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _ddtw_from_multiple_to_multiple_distance(
        _X, _y, window, itakura_max_slope, unequal_length
    )


@njit(cache=True, fastmath=True)
def _ddtw_pairwise_distance(
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

    X_average_of_slope = NumbaList()
    for i in range(n_cases):
        X_average_of_slope.append(average_of_slope(X[i]))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X_average_of_slope[i], X_average_of_slope[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _dtw_distance(x1, x2, bounding_matrix)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _ddtw_from_multiple_to_multiple_distance(
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

    # Derive the arrays before so that we dont have to redo every iteration
    x_average_of_slope = NumbaList()
    for i in range(n_cases):
        x_average_of_slope.append(average_of_slope(x[i]))

    y_average_of_slope = NumbaList()
    for i in range(m_cases):
        y_average_of_slope.append(average_of_slope(y[i]))

    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x_average_of_slope[i], y_average_of_slope[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _dtw_distance(x1, y1, bounding_matrix)
    return distances


@njit(cache=True, fastmath=True)
def ddtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the DDTW alignment path between two time series.

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
        The ddtw distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, or 2D arrays.
        If n_timepoints or m_timepoints are less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import ddtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> ddtw_alignment_path(x, y)
    ([(0, 0), (1, 1)], 0.25)
    """
    cost_matrix = ddtw_cost_matrix(x, y, window, itakura_max_slope)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 3, y.shape[-1] - 3],
    )


@njit(cache=True, fastmath=True)
def average_of_slope(q: np.ndarray) -> np.ndarray:
    r"""Compute the average of a slope between points.

    Computes the average of the slope of the line through the point in question and
    its left neighbour, and the slope of the line through the left neighbour and the
    right neighbour. proposed in [1] for use in this context.
    .. math::
    q'_(i) = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}
    Where q is the original time series and q' is the derived time series.

    Parameters
    ----------
    q : np.ndarray (n_channels, n_timepoints)
        Time series to take derivative of.

    Returns
    -------
    np.ndarray (n_channels, n_timepoints - 2)
        Array containing the derivative of q.

    Raises
    ------
    ValueError
        If the time series has less than 3 points.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances.elastic._ddtw import average_of_slope
    >>> q = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> average_of_slope(q)
    array([[1., 1., 1., 1., 1., 1., 1., 1.]])
    """
    if q.shape[1] < 3:
        raise ValueError("Time series must have at least 3 points.")
    result = np.zeros((q.shape[0], q.shape[1] - 2))
    for i in range(q.shape[0]):
        for j in range(1, q.shape[1] - 1):
            result[i, j - 1] = (
                (q[i, j] - q[i, j - 1]) + (q[i, j + 1] - q[i, j - 1]) / 2.0
            ) / 2.0
    return result
