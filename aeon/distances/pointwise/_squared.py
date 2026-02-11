"""Squared distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._distance_factory._distance_factory import (
    build_distance,
    build_pairwise_distance,
)


@njit(cache=True, fastmath=True, inline="always")
def _univariate_squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Univariate squared distance for 1D arrays."""
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        diff = x[i] - y[i]
        distance += diff * diff
    return distance


@njit(cache=True, fastmath=True)
def _squared_distance_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Squared distance for 2D inputs (n_channels, n_timepoints)."""
    distance = 0.0
    min_channels = min(x.shape[0], y.shape[0])
    for c in range(min_channels):
        distance += _univariate_squared_distance(x[c], y[c])
    return distance


# Build distance function and copy docstring
squared_distance = build_distance(
    core_distance=_squared_distance_2d,
    name="squared",
)

squared_distance.__doc__ = """Compute the squared distance between two time series.

The squared distance between two time series is defined as:

.. math::
    sd(x, y) = \\sum_{i=1}^{n} (x_i - y_i)^2

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
    Squared distance between x and y.

Raises
------
ValueError
    If x and y are not 1D or 2D arrays.

Examples
--------
>>> import numpy as np
>>> from aeon.distances import squared_distance
>>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
>>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
>>> squared_distance(x, y)
1000.0
"""

squared_pairwise_distance = build_pairwise_distance(
    core_distance=squared_distance,
    name="squared",
)

squared_pairwise_distance.__doc__ = """Compute the squared pairwise distance.

Parameters
----------
X : np.ndarray or List of np.ndarray
    A collection of time series instances  of shape
    ``(n_cases, n_timepoints)`` or ``(n_cases, n_channels, n_timepoints)``.
y : np.ndarray or List of np.ndarray or None, default=None
    A single series or a collection of time series of shape
    ``(m_timepoints,)`` or ``(m_cases, m_timepoints)`` or
    ``(m_cases, m_channels, m_timepoints)``.
    If None, then the squared pairwise distance between the instances of X
    is calculated.
n_jobs : int, default=1
    The number of jobs to run in parallel. If -1, then the number of jobs is
    set to the number of CPU cores. If 1, then the function is executed in a
    single thread. If greater than 1, then the function is executed in
    parallel.

    NOTE: For this distance function unless your data has a large number of
    time points, it is recommended to use n_jobs=1.

Returns
-------
np.ndarray (n_cases, n_cases)
    squared pairwise matrix between the instances of X.

Raises
------
ValueError
    If X is not 2D or 3D array when only passing X.
    If X and y are not 1D, 2D or 3D arrays when passing both X and y.

Examples
--------
>>> import numpy as np
>>> from aeon.distances import squared_pairwise_distance
>>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
>>> squared_pairwise_distance(X)
array([[  0.,  28., 109.],
       [ 28.,   0.,  27.],
       [109.,  27.,   0.]])
"""
