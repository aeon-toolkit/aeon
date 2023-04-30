# -*- coding: utf-8 -*-
r"""Weighted dynamic time warping (WDTW) distance between two time series.

WDTW uses DTW with a weighted pairwise distance matrix rather than a window. When
creating the distance matrix :math:'M', a weight penalty  :math:'w_{|i-j|}' for a
warping distance of :math:'|i-j|' is applied, so that for series
:math:'a = <a_1, ..., a_m>' and :math:'b=<b_1,...,b_m>',
.. math:: M_{i,j}=  w(|i-j|) (a_i-b_j)^2.
A logistic weight function, proposed in [1] is used, so that a warping of :math:'x'
places imposes a weighting of
.. math:: w(x)=\frac{w_{max}}{1+e^{-g(x-m/2)}},
where :math:'w_{max}' is an upper bound on the weight (set to 1), :math:'m' is
the series length and :math:'g' is a parameter that controls the penalty level
for larger warpings. The greater :math:'g' is, the greater the penalty for warping.
Once :math:'M' is found, standard dynamic time warping is applied.

WDTW is set up so you can use it with a bounding box in addition to the weight
function is so desired. This is for consistency with the other distance functions.

References
----------
.. [1] Jeong, Y., Jeong, M., Omitaomu, O.: Weighted dynamic time warping for time
series classification. Pattern Recognition 44, 2231–2240 (2011)
"""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance


@njit(cache=True, fastmath=True)
def wdtw_distance(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> float:
    """Compute the wdtw distance between two time series.

    First proposed in [1]_, WDTW adds a  adds a multiplicative weight penalty based on
    the warping distance. This means that time series with lower phase difference have
    a smaller weight imposed (i.e less penalty imposed) and time series with larger
    phase difference have a larger weight imposed (i.e. larger penalty imposed).

    Formally this can be described as:

    .. math::
        d_{w}(x_{i}, y_{j}) = ||w_{|i-j|}(x_{i} - y_{j})||

    Where d_w is the distance with a the weight applied to it for points i, j, where
    w(|i-j|) is a positive weight between the two points x_i and y_j.

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
        wdtw distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wdtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> wdtw_distance(x, y)
    356.5017799592854

    References
    ----------
    .. [1] Young-Seon Jeong, Myong K. Jeong, Olufemi A. Omitaomu, Weighted dynamic time
    warping for time series classification, Pattern Recognition, Volume 44, Issue 9,
    2011, Pages 2231-2240, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2010.09.022.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _wdtw_distance(_x, _y, bounding_matrix, g)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _wdtw_distance(x, y, bounding_matrix, g)
    if x.ndim == 3 and y.ndim == 3:
        distance = 0
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        for curr_x, curr_y in zip(x, y):
            distance += _wdtw_distance(curr_x, curr_y, bounding_matrix, g)
        return distance
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def wdtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> np.ndarray:
    """Compute the wdtw cost matrix between two time series.

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
        wdtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

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
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _wdtw_cost_matrix(_x, _y, bounding_matrix, g)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _wdtw_cost_matrix(x, y, bounding_matrix, g)
    if x.ndim == 3 and y.ndim == 3:
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        cost_matrix = np.zeros((x.shape[2], y.shape[2]))
        for curr_x, curr_y in zip(x, y):
            cost_matrix = np.add(
                cost_matrix, _wdtw_cost_matrix(curr_x, curr_y, bounding_matrix, g)
            )
        return cost_matrix
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


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


@njit(cache=True, fastmath=True)
def wdtw_pairwise_distance(
    X: np.ndarray, y: np.ndarray = None, window: float = None, g: float = 0.05
) -> np.ndarray:
    """Compute the wdtw pairwise distance between a set of time series.

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

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        wdtw pairwise matrix between the instances of X.

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
    array([[  0.,  26., 108.],
           [ 26.,   0.,  26.],
           [108.,  26.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> wdtw_pairwise_distance(X, y)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> wdtw_pairwise_distance(X, y_univariate)
    array([[300.],
           [147.],
           [ 48.]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _wdtw_pairwise_distance(X, window, g)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _wdtw_pairwise_distance(_X, window, g)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _wdtw_from_multiple_to_multiple_distance(X, y, window, g)
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _wdtw_from_multiple_to_multiple_distance(_x, _y, window, g)
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _wdtw_from_multiple_to_multiple_distance(_x, _y, window, g)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        # Single to multiple
        if X.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return _wdtw_from_multiple_to_multiple_distance(X, _y, window, g)
        if y.ndim == 3 and X.ndim == 2:
            _x = X.reshape((1, X.shape[0], X.shape[1]))
            return _wdtw_from_multiple_to_multiple_distance(_x, y, window, g)
        if X.ndim == 2 and y.ndim == 1:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _wdtw_from_multiple_to_multiple_distance(_x, _y, window, g)
        if y.ndim == 2 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _wdtw_from_multiple_to_multiple_distance(_x, _y, window, g)
        else:
            raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _wdtw_pairwise_distance(X: np.ndarray, window: float, g: float) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _wdtw_distance(X[i], X[j], bounding_matrix, g)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _wdtw_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray, window: float, g: float
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _wdtw_distance(x[i], y[j], bounding_matrix, g)
    return distances


@njit(cache=True, fastmath=True)
def wdtw_alignment_path(
    x: np.ndarray, y: np.ndarray, window: float = None, g: float = 0.05
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the wdtw alignment path between two time series.

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
        The wdtw distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wdtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> wdtw_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 1.90008325008424)
    """
    cost_matrix = wdtw_cost_matrix(x, y, window, g)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
