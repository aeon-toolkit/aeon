"""Helper and common function for similarity search distance profiles."""

__maintainer__ = ["baraline"]

import numpy as np
from scipy.signal import convolve


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
    out : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        Sliding dot product between q and X.
    """
    n_channels, n_timepoints = X.shape
    query_length = q.shape[1]
    out = np.zeros((n_channels, n_timepoints - query_length + 1))
    for i in range(n_channels):
        out[i, :] = convolve(np.flipud(q[i, :]), X[i, :], mode="valid").real
    return out


def get_ith_products(X, T, L, ith):
    """
    Compute dot products between X and the i-th subsequence of size L in T.

    Parameters
    ----------
    X : array, shape = (n_channels, n_timepoints_X)
        Input data.
    T : array, shape =  (n_channels, n_timepoints_T)
        Data containing the query.
    L : int
        Overall query length.
    ith : int
        Query starting index in T.

    Returns
    -------
    np.ndarray, 2D array of shape (n_channels, n_timepoints_X - L + 1)
        Sliding dot product between the i-th subsequence of size L in T and X.

    """
    return fft_sliding_dot_product(X, T[:, ith : ith + L])
