"""Undifferencing Code."""

import numpy as np
from numba import njit

# Needs to be move to DifferenceTransformer at some point


@njit(cache=True, fastmath=True)
def _comb(n, k):
    """
    Calculate the binomial coefficient C(n, k) = n! / (k! * (n - k)!).

    Parameters
    ----------
    n : int
        The total number of items.
    k : int
        The number of items to choose.

    Returns
    -------
    int
        The binomial coefficient C(n, k).
    """
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k  # Take advantage of symmetry
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


@njit(cache=True, fastmath=True)
def _undifference(diff, initial_values):
    """
    Reconstruct original time series from an n-th order differenced series.

    Parameters
    ----------
    diff : array-like
        n-th order differenced series of length N - n
    initial_values : array-like
        The first n values of the original series before differencing (length n)

    Returns
    -------
    original : np.ndarray
        Reconstructed original series of length N
    """
    n = len(initial_values)
    kernel = np.array(
        [(-1) ** (k + 1) * _comb(n, k) for k in range(1, n + 1)],
        dtype=initial_values.dtype,
    )
    original = np.empty((n + len(diff)), dtype=initial_values.dtype)
    original[:n] = initial_values

    for i, d in enumerate(diff):
        original[n + i] = np.dot(kernel, original[i : n + i][::-1]) + d

    return original
