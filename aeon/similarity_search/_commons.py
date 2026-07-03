"""Shared numba kernels for similarity search."""

__maintainer__ = ["baraline"]
__all__ = ["_pairwise_squared_distance"]

import numpy as np
from numba import njit, prange

from aeon.distances import squared_distance


@njit(cache=True, fastmath=True, parallel=True)
def _pairwise_squared_distance(X, Q):
    """
    Compute squared Euclidean distance between Q and each series in X.

    Parameters
    ----------
    X : np.ndarray, shape=(n_cases, n_channels, n_timepoints)
        Collection of time series (or subsequences).
    Q : np.ndarray, shape=(n_channels, n_timepoints)
        Query series.

    Returns
    -------
    distances : np.ndarray, shape=(n_cases,)
        Squared Euclidean distance from Q to each series in X.
    """
    # ``squared_distance`` is a serial @njit on 2D arrays, so calling it inside the
    # prange is macOS-safe (no nested parallelism). It gives identical results and
    # speed to the inlined triple loop; the public ``squared_pairwise_distance`` is
    # not used because its typed-list conversion is 10-27x slower here.
    n = X.shape[0]
    distances = np.zeros(n)
    for i in prange(n):
        distances[i] = squared_distance(X[i], Q)
    return distances
