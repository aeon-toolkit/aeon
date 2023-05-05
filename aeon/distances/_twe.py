# -*- coding: utf-8 -*-
"""Time Warp Edit (TWE) distance between two time series.

The Time Warp Edit (TWE) distance is a distance measure for discrete time series
matching with time 'elasticity'. In comparison to other distance measures, (e.g.
DTW (Dynamic Time Warping) or LCS (Longest Common Subsequence Problem)), TWE is a
metric. Its computational time complexity is O(n^2), but can be drastically reduced
in some specific situation by using a corridor to reduce the search space. Its
memory space complexity can be reduced to O(n). It was first proposed in [1].

References
----------
.. [1] Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment
for Time Series Matching". IEEE Transactions on Pattern Analysis and Machine
Intelligence. 31 (2): 306–318.
"""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import List, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._euclidean import _univariate_euclidean_distance


@njit(cache=True, fastmath=True)
def twe_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
) -> float:
    """Compute the TWE distance between two time series.

    The Time Warp Edit (TWE) distance is a distance measure for discrete time series
    matching with time 'elasticity'. In comparison to other distance measures, (e.g.
    DTW (Dynamic Time Warping) or LCS (Longest Common Subsequence Problem)), TWE is a
    metric. Its computational time complexity is O(n^2), but can be drastically reduced
    in some specific situation by using a corridor to reduce the search space. Its
    memory space complexity can be reduced to O(n). It was first proposed in [1].

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series
    window: int, defaults = None
        Window size. If None, the window size is set to the length of the
        shortest time series.
    nu: float, defaults = 0.001
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    lmbda: float, defaults = 1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.

    Returns
    -------
    float
        TWE distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import twe_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dist = twe_distance(x, y)

    References
    ----------
    .. [1] Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment
    for Time Series Matching". IEEE Transactions on Pattern Analysis and Machine
    Intelligence. 31 (2): 306–318.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _twe_distance(_pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _twe_distance(_pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda)
    if x.ndim == 3 and y.ndim == 3:
        distance = 0
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        for curr_x, curr_y in zip(x, y):
            distance += _twe_distance(
                _pad_arrs(curr_x), _pad_arrs(curr_y), bounding_matrix, nu, lmbda
            )
        return distance
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
) -> np.ndarray:
    """Compute the TWE cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.
    window: int, defaults = None
        Window size. If None, the window size is set to the length of the
        shortest time series.
    nu: float, defaults = 0.001
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    lmbda: float, defaults = 1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        TWE cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import twe_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    >>> twe_cost_matrix(x, y)
    array([[ 0.   ,  2.001,  4.002,  6.003,  8.004, 10.005, 12.006, 14.007],
           [ 2.001,  0.   ,  2.001,  4.002,  6.003,  8.004, 10.005, 12.006],
           [ 4.002,  2.001,  0.   ,  2.001,  4.002,  6.003,  8.004, 10.005],
           [ 6.003,  4.002,  2.001,  0.   ,  2.001,  4.002,  6.003,  8.004],
           [ 8.004,  6.003,  4.002,  2.001,  0.   ,  2.001,  4.002,  6.003],
           [10.005,  8.004,  6.003,  4.002,  2.001,  0.   ,  2.001,  4.002],
           [12.006, 10.005,  8.004,  6.003,  4.002,  2.001,  0.   ,  2.001],
           [14.007, 12.006, 10.005,  8.004,  6.003,  4.002,  2.001,  0.   ]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(_x.shape[1], _y.shape[1], window)
        return _twe_cost_matrix(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(x.shape[1], y.shape[1], window)
        return _twe_cost_matrix(_pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda)
    if x.ndim == 3 and y.ndim == 3:
        bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)
        cost_matrix = np.zeros((x.shape[2], y.shape[2]))
        for curr_x, curr_y in zip(x, y):
            cost_matrix = np.add(
                cost_matrix,
                _twe_cost_matrix(
                    _pad_arrs(curr_x), _pad_arrs(curr_y), bounding_matrix, nu, lmbda
                ),
            )
        return cost_matrix
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def _twe_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, nu: float, lmbda: float
) -> float:
    return _twe_cost_matrix(x, y, bounding_matrix, nu, lmbda)[
        x.shape[1] - 2, y.shape[1] - 2
    ]


@njit(cache=True, fastmath=True)
def _twe_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, nu: float, lmbda: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                # Deletion in x
                del_x_squared_dist = _univariate_euclidean_distance(
                    x[:, i - 1], x[:, i]
                )
                del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add
                # Deletion in y
                del_y_squared_dist = _univariate_euclidean_distance(
                    y[:, j - 1], y[:, j]
                )
                del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add

                # Match
                match_same_squared_d = _univariate_euclidean_distance(x[:, i], y[:, j])
                match_prev_squared_d = _univariate_euclidean_distance(
                    x[:, i - 1], y[:, j - 1]
                )
                match = (
                    cost_matrix[i - 1, j - 1]
                    + match_same_squared_d
                    + match_prev_squared_d
                    + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )

                cost_matrix[i, j] = min(del_x, del_y, match)

    return cost_matrix[1:, 1:]


@njit(cache=True, fastmath=True)
def _pad_arrs(x: np.ndarray) -> np.ndarray:
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1))
    zero_arr = np.array([0.0])
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x


@njit(cache=True, fastmath=True)
def twe_pairwise_distance(
    X: np.ndarray,
    y: np.ndarray = None,
    window: float = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
) -> np.ndarray:
    """Compute the twe pairwise distance between a set of time series.

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
    nu: float, defaults = 0.001
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    lmbda: float, defaults = 1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        twe pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import twe_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> twe_pairwise_distance(X)
    array([[ 0.   , 11.004, 14.004],
           [11.004,  0.   , 11.004],
           [14.004, 11.004,  0.   ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> twe_pairwise_distance(X, y)
    array([[18.004, 21.004, 24.004],
           [15.004, 18.004, 21.004],
           [12.004, 15.004, 18.004]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> twe_pairwise_distance(X, y_univariate)
    array([[19.46810162],
           [16.46810162],
           [13.46810162]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _twe_pairwise_distance(X, window, nu, lmbda)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _twe_pairwise_distance(_X, window, nu, lmbda)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _twe_from_multiple_to_multiple_distance(X, y, window, nu, lmbda)
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _twe_from_multiple_to_multiple_distance(_x, _y, window, nu, lmbda)
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _twe_from_multiple_to_multiple_distance(_x, _y, window, nu, lmbda)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        # Single to multiple
        if X.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return _twe_from_multiple_to_multiple_distance(X, _y, window, nu, lmbda)
        if y.ndim == 3 and X.ndim == 2:
            _x = X.reshape((1, X.shape[0], X.shape[1]))
            return _twe_from_multiple_to_multiple_distance(_x, y, window, nu, lmbda)
        if X.ndim == 2 and y.ndim == 1:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _twe_from_multiple_to_multiple_distance(_x, _y, window, nu, lmbda)
        if y.ndim == 2 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _twe_from_multiple_to_multiple_distance(_x, _y, window, nu, lmbda)
        else:
            raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _twe_pairwise_distance(
    X: np.ndarray, window: float, nu: float, lmbda: float
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(X.shape[2], X.shape[2], window)

    # Pad the arrays before so that we don't have to redo every iteration
    padded_X = np.zeros((X.shape[0], X.shape[1], X.shape[2] + 1))
    for i in range(X.shape[0]):
        padded_X[i] = _pad_arrs(X[i])

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _twe_distance(
                padded_X[i], padded_X[j], bounding_matrix, nu, lmbda
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _twe_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float,
    nu: float,
    lmbda: float,
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(x.shape[2], y.shape[2], window)

    # Pad the arrays before so that we dont have to redo every iteration
    padded_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 1))
    for i in range(x.shape[0]):
        padded_x[i] = _pad_arrs(x[i])

    padded_y = np.zeros((y.shape[0], y.shape[1], y.shape[2] + 1))
    for i in range(y.shape[0]):
        padded_y[i] = _pad_arrs(y[i])

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _twe_distance(
                padded_x[i], padded_y[j], bounding_matrix, nu, lmbda
            )
    return distances


@njit(cache=True, fastmath=True)
def twe_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the twe alignment path between two time series.

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
    nu: float, defaults = 0.001
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    lmbda: float, defaults = 1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The twe distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import twe_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> twe_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 2.0)
    """
    bounding_matrix = create_bounding_matrix(x.shape[-1], y.shape[-1], window)
    cost_matrix = twe_cost_matrix(x, y, window, nu, lmbda)
    # Need to do this because the cost matrix contains 0s and not inf in out of bounds
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
