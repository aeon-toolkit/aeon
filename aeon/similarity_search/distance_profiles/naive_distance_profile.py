"""Naive distance profile computation."""

__author__ = ["baraline"]

import numpy as np
from numba import njit

from aeon.similarity_search.distance_profiles._commons import (
    AEON_SIMSEARCH_STD_THRESHOLD,
    _get_input_sizes,
    _z_normalize_1D_series_with_mean_std,
    _z_normalize_2D_series_with_mean_std,
)


def naive_distance_profile(X, q, mask, numba_distance_function):
    r"""
    Compute a distance profile in a brute force way.

    It computes the distance profiles between the input time series and the query using
    the specified distance. The search is made in a brute force way without any
    optimizations and can thus be slow.

    A distance profile between a (univariate) time series :math:`X_i = {x_1, ..., x_m}`
    and a query :math:`Q = {q_1, ..., q_m}` is defined as a vector of size :math:`m-(
    l-1)`, such as :math:`P(X_i, Q) = {d(C_1, Q), ..., d(C_m-(l-1), Q)}` with d the
    distance function, and :math:`C_j = {x_j, ..., x_{j+(l-1)}}` the j-th candidate
    subsequence of size :math:`l` in :math:`X_i`.

    Parameters
    ----------
    X: array shape (n_cases, n_channels, series_length)
        The input samples.
    q : np.ndarray shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_instances, n_channels, n_timestamps - (q_length - 1))
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.
    numba_distance_function : func
        A numba njit function used to compute the distance between two 1D vectors.

    Returns
    -------
    distance_profile : np.ndarray
        shape (n_cases, n_channels, series_length - query_length + 1)
        The distance profile between q and the input time series X independently
        for each channel.

    """
    return _naive_distance_profile(X, q, mask, numba_distance_function)


def normalized_naive_distance_profile(
    X, q, mask, X_means, X_stds, q_means, q_stds, numba_distance_function
):
    """
    Compute a distance profile in a brute force way.

    It computes the distance profiles between the input time series and the query using
    the specified distance. The search is made in a brute force way without any
    optimizations and can thus be slow.

    A distance profile between a (univariate) time series :math:`X_i = {x_1, ..., x_m}`
    and a query :math:`Q = {q_1, ..., q_m}` is defined as a vector of size :math:`m-(
    l-1)`, such as :math:`P(X_i, Q) = {d(C_1, Q), ..., d(C_m-(l-1), Q)}` with d the
    distance function, and :math:`C_j = {x_j, ..., x_{j+(l-1)}}` the j-th candidate
    subsequence of size :math:`l` in :math:`X_i`.

    Parameters
    ----------
    X : array, shape (n_instances, n_channels, series_length)
        The input samples.
    q : array, shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_instances, n_channels, n_timestamps - (q_length - 1))
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.
    X_means : array, shape (n_instances, n_channels, series_length - (query_length-1))
        Means of each subsequences of X of size query_length
    X_stds : array, shape (n_instances, n_channels, series_length - (query_length-1))
        Stds of each subsequences of X of size query_length
    q_means : array, shape (n_channels)
        Means of the query q
    q_stds : array, shape (n_channels)
        Stds of the query q
     numba_distance_function : func
         A numba njit function used to compute the distance between two 1D vectors.

    Returns
    -------
    distance_profile : np.ndarray
        shape (n_instances, n_channels, series_length - query_length + 1).
        The distance profile between q and the input time series X independently
        for each channel.

    """
    # Make STDS inferior to the threshold to 1 to avoid division per 0 error.
    q_stds[q_stds < AEON_SIMSEARCH_STD_THRESHOLD] = 1
    X_stds[X_stds < AEON_SIMSEARCH_STD_THRESHOLD] = 1
    return _normalized_naive_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds, numba_distance_function
    )


@njit(cache=True, fastmath=True)
def _naive_distance_profile(X, q, mask, numba_distance_function):
    n_instances, n_channels, X_length, q_length, profile_size = _get_input_sizes(X, q)
    distance_profile = np.full((n_instances, n_channels, profile_size), np.inf)

    for i_instance in range(n_instances):
        for i_channel in range(n_channels):
            for i_candidate in range(profile_size):
                if mask[i_instance, i_channel, i_candidate]:
                    distance_profile[
                        i_instance, i_channel, i_candidate
                    ] = numba_distance_function(
                        q[i_channel],
                        X[i_instance, i_channel, i_candidate : i_candidate + q_length],
                    )

    return distance_profile


@njit(cache=True, fastmath=True)
def _normalized_naive_distance_profile(
    X, q, mask, X_means, X_stds, q_means, q_stds, numba_distance_function
):
    n_instances, n_channels, X_length, q_length, profile_size = _get_input_sizes(X, q)
    q = _z_normalize_2D_series_with_mean_std(q, q_means, q_stds)
    distance_profile = np.full((n_instances, n_channels, profile_size), np.inf)

    # Compute euclidean distance for all candidate in a "brute force" way
    for i_instance in range(n_instances):
        for i_channel in range(n_channels):
            for i_candidate in range(profile_size):
                if mask[i_instance, i_channel, i_candidate]:
                    # Extract and normalize the candidate
                    _C = _z_normalize_1D_series_with_mean_std(
                        X[i_instance, i_channel, i_candidate : i_candidate + q_length],
                        X_means[i_instance, i_channel, i_candidate],
                        X_stds[i_instance, i_channel, i_candidate],
                    )
                    distance_profile[
                        i_instance, i_channel, i_candidate
                    ] = numba_distance_function(q[i_channel], _C)
    return distance_profile
