__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def minkowski_distance(
    x: np.ndarray, y: np.ndarray, p: float = 2.0, w: Optional[np.ndarray] = None
) -> float:
    r"""Compute the Minkowski distance between two time series.

    The Minkowski distance between two time series of length m
    with a given parameter p is defined as:
    .. math::
        md(x, y, p) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}

    Optionally, a weight vector w can be provided to
    give different weights to the elements:
    .. math::
        md_w(x, y, p, w) = \left( \sum_{i=1}^{n} w_i \cdot |x_i - y_i|^p \right)^{\frac{1}{p}} # noqa: E501

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    p : float, default=2.0
        The order of the norm of the difference
        (default is 2.0, which represents the Euclidean distance).
    w : np.ndarray, default=None
        An array of weights, of the same shape as x and y
        (default is None, which implies equal weights).

    Returns
    -------
    float
        Minkowski distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If p is less than 1.
        If w is provided and its shape does not match x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import minkowski_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> minkowski_distance(x, y, p=1)
    100.0

    >>> x = np.array([1, 0, 0])
    >>> y = np.array([0, 1, 0])
    >>> w = np.array([2,2,2])
    >>> minkowski_distance(x, y, p=2, w=w) # doctest: +SKIP
    2.0
    """
    if x.ndim not in (1, 2) or y.ndim not in (1, 2):
        raise ValueError("x and y must be 1D or 2D arrays")

    if p < 1:
        raise ValueError("p should be greater or equal to 1")

    # Handle Weight
    if w is not None:
        _w = w.astype(x.dtype)
        if x.shape != _w.shape:
            raise ValueError("Weights w must have the same shape as x")
        if np.any(_w < 0):
            raise ValueError("Input weights should be all non-negative")
    else:
        _w = np.ones_like(x)

    if x.ndim == 1 and y.ndim == 1:
        return _univariate_minkowski_distance(x, y, p, _w)
    if x.ndim == 2 and y.ndim == 2:
        return _multivariate_minkowski_distance(x, y, p, _w)

    raise ValueError("Inconsistent dimensions.")


@njit(cache=True, fastmath=True)
def _univariate_minkowski_distance(
    x: np.ndarray, y: np.ndarray, p: float, w: np.ndarray
) -> float:
    min_length = min(x.shape[0], y.shape[0])

    x = x[:min_length]
    y = y[:min_length]
    w = w[:min_length]

    dist = np.sum(w * (np.abs(x - y) ** p))

    return float(dist ** (1.0 / p))


@njit(cache=True, fastmath=True)
def _multivariate_minkowski_distance(
    x: np.ndarray, y: np.ndarray, p: float, w: np.ndarray
) -> float:
    dist = 0.0
    min_rows = min(x.shape[0], y.shape[0])

    for i in range(min_rows):
        min_cols = min(x[i].shape[0], y[i].shape[0])
        x_row = x[i][:min_cols]
        y_row = y[i][:min_cols]
        w_row = w[i][:min_cols]

        diff = np.abs(x_row - y_row) ** p
        dist += np.sum(w_row * diff)

    return dist ** (1.0 / p)


def minkowski_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    p: float = 2.0,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the Minkowski pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the minkoski pairwise distance between the instances of X is
        calculated.
    p : float, default=2.0
        The order of the norm of the difference
        (default is 2.0, which represents the Euclidean distance).
    w : np.ndarray, default=None
        An array of weights, applied to each pairwise calculation.
        The weights should match the shape of the time series in X and y.

    Returns
    -------
    np.ndarray
        Minkowski pairwise distance matrix between
        the instances of X (and y if provided).

    Raises
    ------
    ValueError
        If X (and y if provided) are not 1D, 2D or 3D arrays.
        If p is less than 1.
        If w is provided and its shape does not match the instances in X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import minkowski_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> minkowski_pairwise_distance(X, p=1)
    array([[ 0., 10., 19.],
           [10.,  0.,  9.],
           [19.,  9.,  0.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> minkowski_pairwise_distance(X, y,p=2)
    array([[17.32050808, 22.5166605 , 27.71281292],
           [12.12435565, 17.32050808, 22.5166605 ],
           [ 6.92820323, 12.12435565, 17.32050808]])

    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([[11, 12, 13], [14, 15, 16]])
    >>> w = np.array([[21, 22, 23], [24, 25, 26]])
    >>> minkowski_pairwise_distance(X, y, p=2, w=w)
    array([[ 81.24038405, 105.61249926],
           [ 60.62177826,  86.60254038]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> minkowski_pairwise_distance(X, y_univariate, p=1)
    array([[30.],
           [21.],
           [12.]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> minkowski_pairwise_distance(X)
    array([[ 0.        ,  5.19615242, 12.12435565],
           [ 5.19615242,  0.        ,  8.        ],
           [12.12435565,  8.        ,  0.        ]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, _ = _convert_collection_to_numba_list(X, "X", multivariate_conversion)
    if y is None:
        return _minkowski_pairwise_distance(_X, p, w)

    _y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
    return _minkowski_from_multiple_to_multiple_distance(_X, _y, p, w)


@njit(cache=True, fastmath=True)
def _minkowski_pairwise_distance(
    X: NumbaList[np.ndarray], p: float, w: Optional[np.ndarray] = None
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            if w is None:
                distances[i, j] = minkowski_distance(X[i], X[j], p)
            else:
                # Reshape weights to 2D for matching instance
                # dimensions in distance calculation.
                _w = w[i].reshape((1, w.shape[1]))
                distances[i, j] = minkowski_distance(X[i], X[j], p, _w)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _minkowski_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    p: float,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            if w is None:
                distances[i, j] = minkowski_distance(x[i], y[j], p)
            else:
                _w = w[i].reshape((1, w.shape[1]))
                distances[i, j] = minkowski_distance(x[i], y[j], p, _w)

    return distances
