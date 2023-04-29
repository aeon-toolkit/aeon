# -*- coding: utf-8 -*-
r"""Derivative Dynamic Time Warping (DDTW) distance.

DDTW is an adaptation of DTW originally proposed in [1]_. DDTW attempts to
improve on dtw by better account for the 'shape' of the time series.
This is done by considering y axis data points as higher level features of 'shape'.
To do this the first derivative of the sequence is taken, and then using this
derived sequence a dtw computation is done.
The default derivative used is:

.. math::
    D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

Where q is the original time series and d_q is the derived time series.

References
----------
.. [1] Keogh, Eamonn & Pazzani, Michael. (2002). Derivative Dynamic Time Warping.
    First SIAM International Conference on Data Mining.
    1. 10.1137/1.9781611972719.1.
"""
__author__ = ["chrisholder", "tonybagnall"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._dtw import _dtw_cost_matrix, _dtw_distance, create_bounding_matrix


@njit(cache=True, fastmath=True)
def ddtw_distance(x: np.ndarray, y: np.ndarray, window: float = None) -> float:
    r"""Compute the ddtw distance between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    float
        ddtw distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.
        If n_timepoints or m_timepoints are less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import ddtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[42, 23, 21, 55, 1, 19, 33, 34, 29, 19]])
    >>> ddtw_distance(x, y)
    2179.9375
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = average_of_slope(x.reshape((1, x.shape[0])))
        _y = average_of_slope(y.reshape((1, y.shape[0])))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _dtw_distance(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        _x = average_of_slope(x)
        _y = average_of_slope(y)
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _dtw_distance(_x, _y, bounding_matrix)
    if x.ndim == 3 and y.ndim == 3:
        distance = 0
        bounding_matrix = create_bounding_matrix(x.shape[2] - 2, y.shape[2] - 2, window)
        for curr_x, curr_y in zip(x, y):
            _x = average_of_slope(curr_x)
            _y = average_of_slope(curr_y)
            distance += _dtw_distance(_x, _y, bounding_matrix)
        return distance
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def ddtw_cost_matrix(x: np.ndarray, y: np.ndarray, window: float = None) -> np.ndarray:
    r"""Compute the ddtw cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        ddtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.
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
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _dtw_cost_matrix(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        _x = average_of_slope(x)
        _y = average_of_slope(y)
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _dtw_cost_matrix(_x, _y, bounding_matrix)
    if x.ndim == 3 and y.ndim == 3:
        bounding_matrix = create_bounding_matrix(x.shape[2] - 2, y.shape[2] - 2, window)
        cost_matrix = np.zeros((x.shape[2] - 2, y.shape[2] - 2))
        for curr_x, curr_y in zip(x, y):
            _x = average_of_slope(curr_x)
            _y = average_of_slope(curr_y)
            cost_matrix = np.add(cost_matrix, _dtw_cost_matrix(_x, _y, bounding_matrix))
        return cost_matrix
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def ddtw_pairwise_distance(
    X: np.ndarray, y: np.ndarray = None, window: float = None
) -> np.ndarray:
    """Compute the ddtw pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints)
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,), default=None
        A collection of time series instances.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
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
    >>> y_univariate = np.array([[100, 11, 199],[10, 15, 26], [170, 108, 1119]])
    >>> ddtw_pairwise_distance(X, y_univariate)
    array([[ 15129.    ],
           [322624.    ],
           [  3220.5625]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _ddtw_pairwise_distance(X, window)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _ddtw_pairwise_distance(_X, window)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _ddtw_from_multiple_to_multiple_distance(X, y, window)
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _ddtw_from_multiple_to_multiple_distance(_x, _y, window)
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _ddtw_from_multiple_to_multiple_distance(_x, _y, window)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        # Single to multiple
        if X.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return _ddtw_from_multiple_to_multiple_distance(X, _y, window)
        if y.ndim == 3 and X.ndim == 2:
            _x = X.reshape((1, X.shape[0], X.shape[1]))
            return _ddtw_from_multiple_to_multiple_distance(_x, y, window)
        if X.ndim == 2 and y.ndim == 1:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _ddtw_from_multiple_to_multiple_distance(_x, _y, window)
        if y.ndim == 2 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _ddtw_from_multiple_to_multiple_distance(_x, _y, window)
        else:
            raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _ddtw_pairwise_distance(X: np.ndarray, window: float) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2] - 2, X.shape[2] - 2, window)

    X_average_of_slope = np.zeros((n_instances, X.shape[1], X.shape[2] - 2))
    for i in range(n_instances):
        X_average_of_slope[i] = average_of_slope(X[i])

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _dtw_distance(
                X_average_of_slope[i], X_average_of_slope[j], bounding_matrix
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _ddtw_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    # Derive the arrays before so that we dont have to redo every iteration
    derive_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] - 2))
    for i in range(x.shape[0]):
        derive_x[i] = average_of_slope(x[i])

    derive_y = np.zeros((y.shape[0], y.shape[1], y.shape[2] - 2))
    for i in range(y.shape[0]):
        derive_y[i] = average_of_slope(y[i])

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _dtw_distance(derive_x[i], derive_y[j], bounding_matrix)
    return distances


@njit(cache=True, fastmath=True)
def ddtw_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the ddtw alignment path between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

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
        If x and y are not 1D, 2D, or 3D arrays.
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
    cost_matrix = ddtw_cost_matrix(x, y, window)
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
    q: np.ndarray (n_channels, n_timepoints)
        Time series to take derivative of.

    Returns
    -------
    np.ndarray  (n_channels, n_timepoints - 2)
        Array containing the derivative of q.

    Raises
    ------
    ValueError
        If the time series has less than 3 points.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances._ddtw import average_of_slope
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
