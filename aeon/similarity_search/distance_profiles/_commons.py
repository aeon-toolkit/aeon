"""Helper and common function for similarity search distance profiles."""

import numpy as np
from numba import njit
from scipy.signal import convolve


@njit(cache=True)
def _get_input_sizes(X, q):
    """
    Get sizes of the input and search space for similarity search.

    Parameters
    ----------
    X : array, shape (n_cases, n_channels, n_timepoints)
         The input samples.
    q : array, shape (n_channels, query_length)
        The input query

    Returns
    -------
    n_cases : int
        Number of samples in X.
    n_channels : int
        Number of channels in X.
    X_length : int
        Number of timestamps in X.
    query_length : int
        Number of timestamps in q
    profile_size : int
        Size of the search space for similarity search for each sample in X

    """
    n_cases, n_channels, n_timepoints = X.shape
    query_length = q.shape[-1]
    profile_size = n_timepoints - query_length + 1
    return (n_cases, n_channels, n_timepoints, query_length, profile_size)


def fft_sliding_dot_product(X, q):
    """
    Use FFT convolution to calculate the sliding window dot product.

    Parameters
    ----------
    X : array, shape=(n_channels, n_timepoints)
        Input time series

    q : array, shape=(n_channels, query_length)
        Input query

    Returns
    -------
    output : shape=(n_channels, n_timepoints - query_length + 1)
        Sliding dot product between q and X.
    """
    n_channels, n_timepoints = X.shape
    query_length = q.shape[1]
    out = np.zeros((n_channels, n_timepoints - query_length + 1))
    for i in range(n_channels):
        out[i, :] = convolve(np.flipud(q[i, :]), X[i, :], mode="valid").real
    return out
