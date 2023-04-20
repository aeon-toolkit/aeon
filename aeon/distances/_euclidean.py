# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

import numpy as np
from numba import njit

from aeon.distances._squared import _univariate_squared_distance, squared_distance


@njit(cache=True)
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the euclidean distance between two time series.

    The Euclidean distance between two time series of length m is the square root of
    the squared distance and is defined as:
    .. math::
        ed(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        First time series.
    y: np.ndarray (n_channels, n_timepoints)
        Second time series.

    Returns
    -------
    float
        Euclidean distance between x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import euclidean_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> euclidean_distance(x, y)
    0.0
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


@njit(cache=True)
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(squared_distance(x, y))


@njit(cache=True)
def _univariate_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(_univariate_squared_distance(x, y))


@njit(cache=True)
def euclidean_pairwise_distance(X: np.ndarray) -> np.ndarray:
    """Compute the euclidean pairwise distance between a set of time series.

    Parameters
    ----------
    X: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        euclidean pairwise matrix between the instances of X.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import euclidean_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> euclidean_pairwise_distance(X)
    array([[ 0.        ,  5.29150262, 10.44030651],
           [ 5.29150262,  0.        ,  5.19615242],
           [10.44030651,  5.19615242,  0.        ]])
    """
    if X.ndim == 3:
        return _euclidean_pairwise_distance(X)
    if X.ndim == 2:
        _X = X.reshape((X.shape[1], 1, X.shape[0]))
        return _euclidean_pairwise_distance(_X)

    raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True)
def _euclidean_pairwise_distance(X: np.ndarray) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = euclidean_distance(X[i], X[j])
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True)
def euclidean_from_single_to_multiple_distance(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Compute the euclidean distance between a single time series and multiple.

    Parameters
    ----------
    x: np.ndarray (n_channels, n_timepoints)
        Single time series.
    y: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.

    Returns
    -------
    np.ndarray (n_instances)
        euclidean distance between the collection of instances in y and the time
        series x.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import euclidean_from_single_to_multiple_distance
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> euclidean_from_single_to_multiple_distance(x, y)
    array([ 2.        ,  6.        , 10.81665383])
    """
    if y.ndim == 3 and x.ndim == 2:
        return _euclidean_from_single_to_multiple_distance(x, y)
    if y.ndim == 2 and x.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((y.shape[0], 1, y.shape[1]))
        return _euclidean_from_single_to_multiple_distance(_x, _y)
    else:
        raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True)
def _euclidean_from_single_to_multiple_distance(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    n_instances = y.shape[0]
    distances = np.zeros(n_instances)

    for i in range(n_instances):
        distances[i] = euclidean_distance(x, y[i])

    return distances


@njit(cache=True)
def euclidean_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Compute the euclidean distance between two sets of time series.

    If x and y are the same then you should use euclidean_pairwise_distance.

    Parameters
    ----------
    x: np.ndarray (n_instances, n_channels, n_timepoints)
        A collection of time series instances.
    y: np.ndarray (m_instances, n_channels, n_timepoints)
        A collection of time series instances.

    Returns
    -------
    np.ndarray (n_instances, m_instances)
        euclidean distance between two collections of time series, x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import euclidean_from_multiple_to_multiple_distance
    >>> x = np.array([[[1, 2, 3, 3]],[[4, 5, 6, 9]], [[7, 8, 9, 22]]])
    >>> y = np.array([[[11, 12, 13, 2]],[[14, 15, 16, 1]], [[17, 18, 19, 10]]])
    >>> euclidean_from_multiple_to_multiple_distance(x, y)
    array([[17.34935157, 22.60530911, 28.58321186],
           [14.        , 19.07878403, 22.53885534],
           [21.16601049, 24.24871131, 21.07130751]])
    """
    if y.ndim == 3 and x.ndim == 3:
        return _euclidean_from_multiple_to_multiple_distance(x, y)
    if y.ndim == 2 and x.ndim == 2:
        _x = x.reshape((x.shape[0], 1, x.shape[1]))
        _y = y.reshape((y.shape[0], 1, y.shape[1]))
        return _euclidean_from_multiple_to_multiple_distance(_x, _y)
    if y.ndim == 1 and x.ndim == 1:
        _x = x.reshape((x.shape[0], 1, 1))
        _y = y.reshape((x.shape[0], 1, 1))
        return _euclidean_from_multiple_to_multiple_distance(_x, _y)
    raise ValueError("x and y must be 1D, 2D, or 3D arrays")


@njit(cache=True)
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
