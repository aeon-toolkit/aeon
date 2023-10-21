"""Helper and common function for similarity search distance profiles."""

import numpy as np
from numba import njit
from scipy.signal import convolve

AEON_SIMSEARCH_STD_THRESHOLD = 1e-7


@njit(cache=True)
def _get_input_sizes(X, q):
    """
    Get sizes of the input and search space for similarity search.

    Parameters
    ----------
    X : array, shape (n_instances, n_channels, series_length)
         The input samples.
    q : array, shape (n_channels, series_length)
        The input query

    Returns
    -------
    n_instances : int
        Number of samples in X.
    n_channels : int
        Number of channels in X.
    X_length : int
        Number of timestamps in X.
    q_length : int
        Number of timestamps in q
    profile_size : int
        Size of the search space for similarity search for each sample in X

    """
    n_instances, n_channels, X_length = X.shape
    q_length = q.shape[-1]
    profile_size = X_length - q_length + 1
    return (n_instances, n_channels, X_length, q_length, profile_size)


@njit(fastmath=True, cache=True)
def _z_normalize_2D_series_with_mean_std(X, mean, std, copy=True):
    """
    Z-normalize a 2D series given the mean and std of each channel.

    Parameters
    ----------
    X : array, shape = (n_channels, n_timestamps)
        Input array to normalize.
    mean : array, shape = (n_channels)
        Mean of each channel of X.
    std : array, shape = (n_channels)
        Std of each channel of X.
    copy : bool, optional
        Wheter to copy the input X to avoid modifying the values of the array it refers
        to (if it is a reference). The default is True.

    Returns
    -------
    X : array, shape = (n_channels, n_timestamps)
        The normalized array
    """
    if copy:
        X = X.copy()
    for i_channel in range(X.shape[0]):
        X[i_channel] = (X[i_channel] - mean[i_channel]) / std[i_channel]
    return X


@njit(fastmath=True, cache=True)
def _z_normalize_1D_series_with_mean_std(X, mean, std, copy=True):
    """
    Z-normalize a 2D series given the mean and std of each channel.

    Parameters
    ----------
    X : array, shape = (n_timestamps)
        Input array to normalize.
    mean : float
        Mean of X.
    std : float
        Std of X.
    copy : bool, optional
        Wheter to copy the input X to avoid modifying the values of the array it refers
        to (if it is a reference). The default is True.

    Returns
    -------
    X : array, shape = (n_channels, n_timestamps)
        The normalized array
    """
    if copy:
        X = X.copy()
    X = (X - mean) / std
    return X


def fft_sliding_dot_product(X, q):
    """
    Use FFT convolution to calculate the sliding window dot product.

    Parameters
    ----------
    X : array, shape=(n_features, n_timestamps)
        Input time series

    q : array, shape=(n_features, q_length)
        Input query

    Returns
    -------
    output : shape=(n_features, n_timestamps - (length - 1))
        Sliding dot product between q and X.
    """
    n_features, n_timestamps = X.shape[0]
    length = q.shape[1]
    out = np.zeros((n_features, n_timestamps - (length - 1)))
    for i in range(n_features):
        out[i, :] = convolve(np.flipud(q[i, :]), X[i, :], mode="valid").real
    return out
