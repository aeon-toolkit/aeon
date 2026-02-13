"""Minkowski distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._distance_factory._distance_factory import (
    build_distance,
    build_pairwise_distance,
)


@njit(cache=True, fastmath=True)
def _minkowski_distance_2d(
    x: np.ndarray, y: np.ndarray, p: float = 2.0, w: np.ndarray | None = None
) -> float:
    """Minkowski distance for 2D inputs.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, shape ``(n_channels, n_timepoints)``.
    p : float
        The order of the norm (must be >= 1).
    w : np.ndarray or None
        Weights array of same shape as x and y, or None for unweighted.

    Returns
    -------
    float
        Minkowski distance between x and y.
    """
    dist = 0.0
    min_rows = min(x.shape[0], y.shape[0])

    if w is None:
        # Unweighted version
        for i in range(min_rows):
            min_cols = min(x[i].shape[0], y[i].shape[0])
            x_row = x[i][:min_cols]
            y_row = y[i][:min_cols]

            diff = np.abs(x_row - y_row) ** p
            dist += np.sum(diff)
    else:
        # Weighted version
        for i in range(min_rows):
            min_cols = min(x[i].shape[0], y[i].shape[0])
            x_row = x[i][:min_cols]
            y_row = y[i][:min_cols]
            w_row = w[i][:min_cols]

            diff = np.abs(x_row - y_row) ** p
            dist += np.sum(w_row * diff)

    return dist ** (1.0 / p)


@njit(cache=True, fastmath=True, inline="always")
def _univariate_minkowski_distance(
    x: np.ndarray, y: np.ndarray, p: float = 2.0, w: np.ndarray | None = None
) -> float:
    """Minkowski distance for univariate 1D arrays."""
    min_length = min(x.shape[0], y.shape[0])

    x = x[:min_length]
    y = y[:min_length]

    if w is None:
        dist = np.sum(np.abs(x - y) ** p)
    else:
        w = w[:min_length]
        dist = np.sum(w * (np.abs(x - y) ** p))

    return float(dist ** (1.0 / p))


@njit(cache=True, fastmath=True)
def _minkowski_core_wrapper(
    x: np.ndarray, y: np.ndarray, p: float = 2.0, w: np.ndarray | None = None
) -> float:
    """Core Minkowski distance that handles both 1D and 2D inputs."""
    if x.ndim == 1:
        return _univariate_minkowski_distance(x, y, p, w)
    else:
        return _minkowski_distance_2d(x, y, p, w)


minkowski_distance = build_distance(
    core_distance=_minkowski_core_wrapper,
    name="minkowski",
)

minkowski_distance.__doc__ = """Compute the Minkowski distance between two time series.

The Minkowski distance between two time series of length m
with a given parameter p is defined as:
.. math::
    md(x, y, p) = \\left( \\sum_{i=1}^{n} |x_i - y_i|^p \\right)^{\\frac{1}{p}}

Optionally, a weight vector w can be provided to
give different weights to the elements:
.. math::
    md_w(x, y, p, w) = \\left( \\sum_{i=1}^{n} w_i \\cdot |x_i - y_i|^p \\right)^{\\frac{1}{p}} # noqa: E501

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


@njit(cache=True, fastmath=True)
def _minkowski_pairwise_core(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """Wrap core function for pairwise with unweighted version."""
    return _minkowski_core_wrapper(x, y, p, None)


minkowski_pairwise_distance = build_pairwise_distance(
    core_distance=_minkowski_pairwise_core,
    name="minkowski",
)

minkowski_pairwise_distance.__doc__ = """Compute the Minkowski pairwise distance.

Parameters
----------
X : np.ndarray or List of np.ndarray
    A collection of time series instances  of shape
    ``(n_cases, n_timepoints)`` or ``(n_cases, n_channels, n_timepoints)``.
y : np.ndarray or List of np.ndarray or None, default=None
    A single series or a collection of time series of shape
    ``(m_timepoints,)`` or ``(m_cases, m_timepoints)`` or
    ``(m_cases, m_channels, m_timepoints)``.
    If None, then the minkowski pairwise distance between the instances of X
    is calculated.
p : float, default=2.0
    The order of the norm of the difference
    (default is 2.0, which represents the Euclidean distance).
w : np.ndarray, default=None
    An array of weights, applied to each pairwise calculation.
    Note: weights are currently not supported for pairwise distances.
n_jobs : int, default=1
    The number of jobs to run in parallel. If -1, then the number of jobs is
    set to the number of CPU cores. If 1, then the function is executed in a
    single thread. If greater than 1, then the function is executed in
    parallel.

Returns
-------
np.ndarray (n_cases, n_cases)
    Minkowski pairwise distance matrix.

Examples
--------
>>> import numpy as np
>>> from aeon.distances import minkowski_pairwise_distance
>>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
>>> minkowski_pairwise_distance(X, p=1)
array([[ 0., 10., 19.],
       [10.,  0.,  9.],
       [19.,  9.,  0.]])
"""
