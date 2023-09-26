# -*- coding: utf-8 -*-
"""Helper and common function for similarity search distance profiles."""


from numba import njit

INF = 1e12


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
