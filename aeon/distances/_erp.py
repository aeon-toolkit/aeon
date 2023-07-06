# -*- coding: utf-8 -*-
r"""Edit real penalty (erp) distance between two time series.

ERP, first proposed in [1]_, attempts to align time series
by better considering how indexes are carried forward through the cost matrix.
Usually in the dtw cost matrix, if an alignment can't be found the previous value
is carried forward. Erp instead proposes the idea of gaps or sequences of points
that have no matches. These gaps are then punished based on their distance from 'g'.

References
----------
.. [1] Lei Chen and Raymond Ng. 2004. On the marriage of Lp-norms and edit distance.
In Proceedings of the Thirtieth international conference on Very large data bases
 - Volume 30 (VLDB '04). VLDB Endowment, 792–803.
"""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import List, Tuple, Union

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._euclidean import _univariate_euclidean_distance
from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    g: float = 0.0,
    g_arr: np.ndarray = None,
) -> float:
    """Compute the ERP distance between two time series.

    ERP, first proposed in [1]_, attempts to align time series
    by better considering how indexes are carried forward through the cost matrix.
    Usually in the dtw cost matrix, if an alignment can't be found the previous value
    is carried forward. Erp instead proposes the idea of gaps or sequences of points
    that have no matches. These gaps are then punished based on their distance from 'g'.

    The optimal value of g is selected from the range [σ/5, σ], where σ is the
    standard deviation of the training data. When there is > 1 channel, g should
    be a np.ndarray where the nth value is the standard deviation of the nth
    channel.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float.
        The reference value to penalise gaps. The default is 0.
    g_arr: np.ndarray of shape (n_channels), defaults=None
        Numpy array that must be the length of the number of channels in x and y.

    Returns
    -------
    float
        ERP distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import erp_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> dist = erp_distance(x, y)

    References
    ----------
    .. [1] Lei Chen and Raymond Ng. 2004. On the marriage of Lp-norms and edit distance.
    In Proceedings of the Thirtieth international conference on Very large data bases
     - Volume 30 (VLDB '04). VLDB Endowment, 792–803.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _erp_distance(_x, _y, bounding_matrix, g, g_arr)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _erp_distance(x, y, bounding_matrix, g, g_arr)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def erp_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    g: Union[float, np.ndarray] = 0.0,
    g_arr: np.ndarray = None,
) -> np.ndarray:
    """Compute the ERP cost matrix between two time series.

    The optimal value of g is selected from the range [σ/5, σ], where σ is the
    standard deviation of the training data. When there is > 1 channel, g should
    be a np.ndarray where the nth value is the standard deviation of the nth
    channel.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: float, defaults=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float.
        The reference value to penalise gaps. The default is 0.
    g_arr: np.ndarray of shape (n_channels), defaults=None
        Numpy array that must be the length of the number of channels in x and y.

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
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _erp_cost_matrix(_x, _y, bounding_matrix, g, g_arr)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _erp_cost_matrix(x, y, bounding_matrix, g, g_arr)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    g: float,
    g_arr: np.ndarray,
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
    g_arr: np.ndarray,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    gx_distance, x_sum = _precompute_g(x, g, g_arr)
    gy_distance, y_sum = _precompute_g(y, g, g_arr)

    cost_matrix[1:, 0] = x_sum
    cost_matrix[0, 1:] = y_sum

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
    x: np.ndarray, g: float, g_array: np.ndarray
) -> Tuple[np.ndarray, float]:
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


@njit(cache=True, fastmath=True)
def erp_pairwise_distance(
    X: np.ndarray,
    y: np.ndarray = None,
    window: float = None,
    g: float = 0.0,
    g_arr: np.ndarray = None,
) -> np.ndarray:
    """Compute the erp pairwise distance between a set of time series.

    The optimal value of g is selected from the range [σ/5, σ], where σ is the
    standard deviation of the training data. When there is > 1 channel, g should
    be a np.ndarray where the nth value is the standard deviation of the nth
    channel.

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
    g: float.
        The reference value to penalise gaps. The default is 0.
    g_arr: np.ndarray of shape (n_channels), defaults=None
        Numpy array that must be the length of the number of channels in x and y.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        erp pairwise matrix between the instances of X.


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
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> erp_pairwise_distance(X, y_univariate)
    array([[30.],
           [21.],
           [12.]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _erp_pairwise_distance(X, window, g, g_arr)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _erp_pairwise_distance(_X, window, g, g_arr)
        raise ValueError("x and y must be 2D or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _erp_from_multiple_to_multiple_distance(_x, _y, window, g, g_arr)


@njit(cache=True, fastmath=True)
def _erp_pairwise_distance(
    X: np.ndarray,
    window: float,
    g: float,
    g_arr: np.ndarray,
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _erp_distance(X[i], X[j], bounding_matrix, g, g_arr)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _erp_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float,
    g: float,
    g_arr: np.ndarray,
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _erp_distance(x[i], y[j], bounding_matrix, g, g_arr)
    return distances


@njit(cache=True, fastmath=True)
def erp_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    g: float = 0.0,
    g_arr: np.ndarray = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the erp alignment path between two time series.

    The optimal value of g is selected from the range [σ/5, σ], where σ is the
    standard deviation of the training data. When there is > 1 channel, g should
    be a np.ndarray where the nth value is the standard deviation of the nth
    channel.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float.
        The reference value to penalise gaps. The default is 0.
    g_arr: np.ndarray of shape (n_channels), defaults=None
        Numpy array that must be the length of the number of channels in x and y.

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
    bounding_matrix = create_bounding_matrix(x.shape[-1], y.shape[-1], window)
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(
        erp_cost_matrix(x, y, window, g, g_arr), bounding_matrix
    )
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
