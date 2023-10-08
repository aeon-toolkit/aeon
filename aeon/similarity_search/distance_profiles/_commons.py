# -*- coding: utf-8 -*-
"""Helper and common function for similarity search distance profiles."""


from numba import njit

AEON_SIMSEARCH_STD_THRESHOLD = 1e-7


@njit(cache=True)
def _get_input_sizes(X, Q):
    """
    Get sizes of the input and search space for similarity search.

    Parameters
    ----------
    X : array, shape (n_samples, n_channels, series_length)
         The input samples.
    Q : array, shape (n_channels, series_length)
        The input query

    Returns
    -------
    n_samples : int
        Number of samples in X.
    n_channels : int
        Number of channeks in X.
    X_length : int
        Number of timestamps in X.
    q_length : int
        Number of timestamps in Q
    search_space_size : int
        Size of the search space for similarity search for each sample in X

    """
    n_samples, n_channels, X_length = X.shape
    q_length = Q.shape[-1]
    search_space_size = X_length - q_length + 1
    return (n_samples, n_channels, X_length, q_length, search_space_size)


@njit(fastmath=True, cache=True)
def _z_normalize_2D_series_with_mean_std(X, mean, std, copy=True):
    """
    Z-normalize a 2D series given the mean and std of each channel.

    Parameters
    ----------
    X : array, shape = (n_channels, n_timestamps)
        Input array to normalize.
    mean : array, shape = (n_channels)
        Mean of each channel.
    std : array, shape = (n_channels)
        Std of each channel.
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
