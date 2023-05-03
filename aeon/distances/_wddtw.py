# -*- coding: utf-8 -*-
"""Weighted derivative dynamic time warping (wddtw) distance between two series.

Takes the first order derivative, then applies _weighted_cost_matrix to find WDTW
distance.
"""
__author__ = ["chrisholder", "tonybagnall"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._ddtw import average_of_slope
from aeon.distances._wdtw import _wdtw_cost_matrix, _wdtw_distance


@njit(cache=True, fastmath=True)
def wddtw_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> float:
    r"""Compute the wddtw distance between two time series.

    WDDTW was first proposed in [1]_ as an extension of DDTW. By adding a weight
    to the derivative it means the alignment isn't only considering the shape of the
    time series, but also the phase.

    Formally the derivative is calculated as:

    .. math::
        D_{x}[q] = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Therefore a weighted derivative can be calculated using D (the derivative) as:

    .. math::
        d_{w}(x_{i}, y_{j}) = ||w_{|i-j|}(D_{x_{i}} - D_{y_{j}})||

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    float
        wddtw distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.
        If n_timepoints or m_timepoints are less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[42, 23, 21, 55, 1, 19, 33, 34, 29, 19]])
    >>> dist = wddtw_distance(x, y)

    References
    ----------
    .. [1] Young-Seon Jeong, Myong K. Jeong, Olufemi A. Omitaomu, Weighted dynamic time
    warping for time series classification, Pattern Recognition, Volume 44, Issue 9,
    2011, Pages 2231-2240, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2010.09.022.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = average_of_slope(x.reshape((1, x.shape[0])))
        _y = average_of_slope(y.reshape((1, y.shape[0])))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _wdtw_distance(_x, _y, bounding_matrix, g)
    if x.ndim == 2 and y.ndim == 2:
        _x = average_of_slope(x)
        _y = average_of_slope(y)
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _wdtw_distance(_x, _y, bounding_matrix, g)
    if x.ndim == 3 and y.ndim == 3:
        distance = 0
        bounding_matrix = create_bounding_matrix(x.shape[2] - 2, y.shape[2] - 2, window)
        for curr_x, curr_y in zip(x, y):
            _x = average_of_slope(curr_x)
            _y = average_of_slope(curr_y)
            distance += _wdtw_distance(_x, _y, bounding_matrix, g)
        return distance
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def wddtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> np.ndarray:
    """Compute the wddtw cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        wddtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.
        If n_timepoints or m_timepoints are less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wddtw_cost_matrix(x, y)
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
        return _wdtw_cost_matrix(_x, _y, bounding_matrix, g)
    if x.ndim == 2 and y.ndim == 2:
        _x = average_of_slope(x)
        _y = average_of_slope(y)
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _wdtw_cost_matrix(_x, _y, bounding_matrix, g)
    if x.ndim == 3 and y.ndim == 3:
        bounding_matrix = create_bounding_matrix(x.shape[2] - 2, y.shape[2] - 2, window)
        cost_matrix = np.zeros((x.shape[2] - 2, y.shape[2] - 2))
        for curr_x, curr_y in zip(x, y):
            _x = average_of_slope(curr_x)
            _y = average_of_slope(curr_y)
            cost_matrix = np.add(
                cost_matrix, _wdtw_cost_matrix(_x, _y, bounding_matrix, g)
            )
        return cost_matrix
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def wddtw_pairwise_distance(
    X: np.ndarray, y: np.ndarray = None, window: float = None, g: float = 0.05
) -> np.ndarray:
    """Compute the wddtw pairwise distance between a set of time series.

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
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.
        If n_timepoints is less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[49, 58, 61]], [[73, 82, 99]]])
    >>> wddtw_pairwise_distance(X)
    array([[ 0.        , 20.86095125, 49.37503255],
           [20.86095125,  0.        ,  6.04844149],
           [49.37503255,  6.04844149,  0.        ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[19, 12, 39]],[[40, 51, 69]], [[79, 28, 91]]])
    >>> y = np.array([[[110, 15, 123]],[[14, 150, 116]], [[9917, 118, 29]]])
    >>> wddtw_pairwise_distance(X, y)
    array([[1.03345029e+03, 4.17910276e+03, 2.68408251e+07],
           [1.60419481e+03, 3.21952986e+03, 2.69227971e+07],
           [2.33574763e+02, 6.64390438e+03, 2.66663693e+07]])

    >>> X = np.array([[[10, 22, 399]],[[41, 500, 1316]], [[117, 18, 9]]])
    >>> y_univariate = np.array([[100, 11, 199],[10, 15, 26], [170, 108, 1119]])
    >>> wddtw_pairwise_distance(X, y_univariate)
    array([[  7469.9486745 ],
           [159295.70501427],
           [  1590.15378267]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _wddtw_pairwise_distance(X, window, g)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _wddtw_pairwise_distance(_X, window, g)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _wddtw_from_multiple_to_multiple_distance(X, y, window, g)
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _wddtw_from_multiple_to_multiple_distance(_x, _y, window, g)
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _wddtw_from_multiple_to_multiple_distance(_x, _y, window, g)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        # Single to multiple
        if X.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return _wddtw_from_multiple_to_multiple_distance(X, _y, window, g)
        if y.ndim == 3 and X.ndim == 2:
            _x = X.reshape((1, X.shape[0], X.shape[1]))
            return _wddtw_from_multiple_to_multiple_distance(_x, y, window, g)
        if X.ndim == 2 and y.ndim == 1:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _wddtw_from_multiple_to_multiple_distance(_x, _y, window, g)
        if y.ndim == 2 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _wddtw_from_multiple_to_multiple_distance(_x, _y, window, g)
        else:
            raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _wddtw_pairwise_distance(X: np.ndarray, window: float, g: float) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2] - 2, X.shape[2] - 2, window)

    X_average_of_slope = np.zeros((n_instances, X.shape[1], X.shape[2] - 2))
    for i in range(n_instances):
        X_average_of_slope[i] = average_of_slope(X[i])

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _wdtw_distance(
                X_average_of_slope[i], X_average_of_slope[j], bounding_matrix, g
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _wddtw_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float, g: float
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    # Derive the arrays before so that we don't have to redo every iteration
    derive_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] - 2))
    for i in range(x.shape[0]):
        derive_x[i] = average_of_slope(x[i])

    derive_y = np.zeros((y.shape[0], y.shape[1], y.shape[2] - 2))
    for i in range(y.shape[0]):
        derive_y[i] = average_of_slope(y[i])

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _wdtw_distance(
                derive_x[i], derive_y[j], bounding_matrix, g
            )
    return distances


@njit(cache=True, fastmath=True)
def wddtw_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the wddtw alignment path between two time series.

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
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The wddtw distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.
        If n_timepoints or m_timepoints are less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> path, dist = wddtw_alignment_path(x, y)
    >>> path
    [(0, 0), (1, 1)]
    """
    cost_matrix = wddtw_cost_matrix(x, y, window, g)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 3, y.shape[-1] - 3],
    )
