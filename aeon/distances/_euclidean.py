# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "tonybagnall"]

import numpy as np
from numba import njit

from aeon.distances._squared import _univariate_squared_distance, squared_distance


@njit(cache=True, fastmath=True)
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the euclidean distance between two time series.

    The Euclidean distance between two time series of length m is the square root of
    the squared distance and is defined as:
    .. math::
        ed(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,) or
            (n_instances, n_channels, n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,) or
            (m_instances, m_channels, m_timepoints)
        Second time series.

    Returns
    -------
    float
        Euclidean distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D, 2D, or 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import euclidean_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> euclidean_distance(x, y)
    31.622776601683793
    """
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_euclidean_distance(x, y)
    if x.ndim == 2 and y.ndim == 2:
        return _euclidean_distance(x, y)
    if x.ndim == 3 and y.ndim == 3:
        distance = 0
        for curr_x, curr_y in zip(x, y):
            distance += _euclidean_distance(curr_x, curr_y)
        return distance
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True, fastmath=True)
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(squared_distance(x, y))


@njit(cache=True, fastmath=True)
def _univariate_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(_univariate_squared_distance(x, y))


@njit(cache=True, fastmath=True)
def euclidean_pairwise_distance(X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """Compute the euclidean pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints) or (n_timepoints,)
        A collection of time series instances.
    y: np.ndarray, of shape (m_instances, m_channels, m_timepoints) or
            (m_instances, m_timepoints) or (m_timepoints,), default=None
        A collection of time series instances.


    Returns
    -------
    np.ndarray (n_instances, n_instances)
        euclidean pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import euclidean_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> euclidean_pairwise_distance(X)
    array([[ 0.        ,  5.29150262, 10.44030651],
           [ 5.29150262,  0.        ,  5.19615242],
           [10.44030651,  5.19615242,  0.        ]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> euclidean_pairwise_distance(X, y)
    array([[17.32050808, 22.5166605 , 27.71281292],
           [12.12435565, 17.32050808, 22.5166605 ],
           [ 6.92820323, 12.12435565, 17.32050808]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([[11, 12, 13],[14, 15, 16], [17, 18, 19]])
    >>> euclidean_pairwise_distance(X, y_univariate)
    array([[17.32050808],
           [12.12435565],
           [ 6.92820323]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _euclidean_pairwise_distance(X)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _euclidean_pairwise_distance(_X)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _euclidean_from_multiple_to_multiple_distance(X, y)
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _euclidean_from_multiple_to_multiple_distance(_x, _y)
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _euclidean_from_multiple_to_multiple_distance(_x, _y)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        # Single to multiple
        if X.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return _euclidean_from_multiple_to_multiple_distance(X, _y)
        if y.ndim == 3 and X.ndim == 2:
            _x = X.reshape((1, X.shape[0], X.shape[1]))
            return _euclidean_from_multiple_to_multiple_distance(_x, y)
        if X.ndim == 2 and y.ndim == 1:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _euclidean_from_multiple_to_multiple_distance(_x, _y)
        if y.ndim == 2 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _euclidean_from_multiple_to_multiple_distance(_x, _y)
        else:
            raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _euclidean_pairwise_distance(X: np.ndarray) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = euclidean_distance(X[i], X[j])
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _euclidean_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = euclidean_distance(x[i], y[j])
    return distances
