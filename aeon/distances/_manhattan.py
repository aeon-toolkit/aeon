__maintainer__ = []

from typing import Optional

import numpy as np
from numba import njit

from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the manhattan distance between two time series.

    The manhattan distance between two time series is defined as:

    .. math::
        manhattan(x, y) = \sum_{i=1}^{n} |x_i - y_i|

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.


    Returns
    -------
    float
        manhattan distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import manhattan_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> manhattan_distance(x, y)
    100.0
    """
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_manhattan_distance(x, y)
    if x.ndim == 2 and y.ndim == 2:
        return _multivariate_manhattan_distance(x, y)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _multivariate_manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_val = min(x.shape[0], y.shape[0])
    for i in range(min_val):
        distance += _univariate_manhattan_distance(x[i], y[i])
    return distance


@njit(cache=True, fastmath=True)
def _univariate_manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        difference = x[i] - y[i]
        distance += abs(difference)
    return distance


@njit(cache=True, fastmath=True)
def manhattan_pairwise_distance(
    X: np.ndarray, y: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the manhattan pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the manhattan pairwise distance between the instances of X is
        calculated.


    Returns
    -------
    np.ndarray (n_cases, n_cases)
        manhattan pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import manhattan_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> manhattan_pairwise_distance(X)
    array([[ 0., 10., 19.],
           [10.,  0.,  9.],
           [19.,  9.,  0.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> manhattan_pairwise_distance(X, y)
    array([[30., 39., 48.],
           [21., 30., 39.],
           [12., 21., 30.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> manhattan_pairwise_distance(X, y_univariate)
    array([[30.],
           [21.],
           [12.]])

    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _manhattan_pairwise_distance(X)
        elif X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _manhattan_pairwise_distance(_X)
        raise ValueError("X must be 2D or 3D array")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _manhattan_from_multiple_to_multiple_distance(_x, _y)


@njit(cache=True, fastmath=True)
def _manhattan_pairwise_distance(X: np.ndarray) -> np.ndarray:
    n_cases = X.shape[0]
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = manhattan_distance(X[i], X[j])
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _manhattan_from_multiple_to_multiple_distance(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    n_cases = x.shape[0]
    m_cases = y.shape[0]
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = manhattan_distance(x[i], y[j])
    return distances
