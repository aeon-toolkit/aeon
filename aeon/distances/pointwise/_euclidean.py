"""Euclidean distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._distance_factory._distance_factory import (
    build_distance,
    build_pairwise_distance,
)
from aeon.distances.pointwise._squared import _univariate_squared_distance


@njit(cache=True, fastmath=True, inline="always")
def _univariate_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance for univariate 1D arrays."""
    return np.sqrt(_univariate_squared_distance(x, y))


@njit(cache=True, fastmath=True)
def _euclidean_distance_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance for 2D inputs (n_channels, n_timepoints)."""
    squared_dist = 0.0
    min_channels = min(x.shape[0], y.shape[0])
    for c in range(min_channels):
        squared_dist += _univariate_squared_distance(x[c], y[c])
    return np.sqrt(squared_dist)


euclidean_distance = build_distance(
    core_distance=_euclidean_distance_2d,
    name="euclidean",
)

euclidean_distance.__doc__ = """Compute the Euclidean distance between two time series.

The Euclidean distance between two time series of length m is the square root of
the squared distance and is defined as:

.. math::
    ed(x, y) = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}

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
    Euclidean distance between x and y.

Raises
------
ValueError
    If x and y are not 1D or 2D arrays.

Examples
--------
>>> import numpy as np
>>> from aeon.distances import euclidean_distance
>>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
>>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
>>> euclidean_distance(x, y)
31.622776601683793
"""

euclidean_pairwise_distance = build_pairwise_distance(
    core_distance=euclidean_distance,
    name="euclidean",
)

euclidean_pairwise_distance.__doc__ = """Compute the Euclidean pairwise distance between a set of time series.

Parameters
----------
X : np.ndarray or List of np.ndarray
    A collection of time series instances  of shape ``(n_cases, n_timepoints)``
    or ``(n_cases, n_channels, n_timepoints)``.
y : np.ndarray or List of np.ndarray or None, default=None
    A single series or a collection of time series of shape ``(m_timepoints,)`` or
    ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
    If None, then the euclidean pairwise distance between the instances of X is
    calculated.
n_jobs : int, default=1
    The number of jobs to run in parallel. If -1, then the number of jobs is set
    to the number of CPU cores. If 1, then the function is executed in a single
    thread. If greater than 1, then the function is executed in parallel.

Returns
-------
np.ndarray (n_cases, n_cases)
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
