"""Helper and common function for similarity search distance profiles."""

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
    output : shape=(n_channels, n_timepoints - query_length + 1)
        Sliding dot product between q and X.
    """
    n_channels, n_timepoints = X.shape
    query_length = q.shape[1]
    out = np.zeros((n_channels, n_timepoints - query_length + 1))
    for i in range(n_channels):
        out[i, :] = convolve(np.flipud(q[i, :]), X[i, :], mode="valid").real
    return out
