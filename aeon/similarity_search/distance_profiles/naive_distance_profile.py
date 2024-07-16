"""Naive distance profile computation."""

__maintainer__ = []


import numpy as np
from numba import njit, prange
from numba.typed import List

from aeon.utils.numba.general import (
    generate_new_default_njit_func,
    z_normalize_series_2d_with_mean_std,
    z_normalize_series_with_mean_std,
)


def naive_distance_profile(X, q, mask, distance_function, distance_args=None):
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
    X: array shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    q : np.ndarray shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_cases, n_channels, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed. Should be a TypedList if X is
        unequal length.
    distance_function : func
        A python function or a numba njit function used to compute the distance between
        two 1D vectors.
    distance_args : dict, default=None
        Dictionary containing keywords arguments to use for the distance_function

    Returns
    -------
    distance_profiles : np.ndarray
        shape (n_cases, n_channels, n_timepoints - query_length + 1)
        The distance profile between q and the input time series X independently
        for each channel. Returns a TypedList if X is unequal length.

    """
    dist_func = generate_new_default_njit_func(distance_function, distance_args)
    # This will compile the new function and check for errors outside the numba loops
    # Call dtype on X[0] to support unequal length inputs
    dist_func(np.ones(3, dtype=X[0].dtype), np.zeros(3, dtype=X[0].dtype))
    distance_profiles = _naive_distance_profile(X, q, mask, dist_func)
    # If input was not unequal length, convert to 3D np array
    if isinstance(X, np.ndarray):
        distance_profiles = np.asarray(distance_profiles)
    return distance_profiles


def normalized_naive_distance_profile(
    X,
    q,
    mask,
    X_means,
    X_stds,
    q_means,
    q_stds,
    distance_function,
    distance_args=None,
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
    X : array, shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    q : array, shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_cases, n_channels, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed. Should be a TypedList if X is
        unequal length.
    X_means : list, shape (n_cases, n_channels, n_timepoints - query_length + 1)
        Means of each subsequences of X of size query_length. Should be a numba
        TypedList if X is unequal length.
    X_stds : list, shape (n_cases, n_channels, n_timepoints - query_length + 1)
        Stds of each subsequences of X of size query_length. Should be a numba
        TypedList if X is unequal length.
    q_means : array, shape (n_channels)
        Means of the query q
    q_stds : array, shape (n_channels)
        Stds of the query q
    distance_function : func
        A python function or a numba njit function used to compute the distance between
        two 1D vectors.
    distance_args : dict, default=None
        Dictionary containing keywords arguments to use for the distance_function

    Returns
    -------
    distance_profiles : np.ndarray
        shape (n_cases, n_channels, n_timepoints - query_length + 1).
        The distance profile between q and the input time series X independently
        for each channel. Returns a TypedList if X is unequal length.

    """
    dist_func = generate_new_default_njit_func(distance_function, distance_args)
    # This will compile the new function and check for errors outside the numba loops
    # Call dtype on X[0] to support unequal length inputs
    dist_func(np.ones(3, dtype=X[0].dtype), np.zeros(3, dtype=X[0].dtype))
    distance_profiles = _normalized_naive_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds, dist_func
    )
    # If input was not unequal length, convert to 3D np array
    if isinstance(X, np.ndarray):
        distance_profiles = np.asarray(distance_profiles)
    return distance_profiles


@njit(cache=True, fastmath=True, parallel=True)
def _naive_distance_profile(
    X,
    q,
    mask,
    numba_distance_function,
):
    distance_profiles = List()
    query_length = q.shape[1]
    n_channels = q.shape[0]

    # Init distance profile array with unequal length support
    for i_instance in range(len(X)):
        profile_length = X[i_instance].shape[1] - query_length + 1
        distance_profiles.append(np.full((n_channels, profile_length), np.inf))

    # Compute distances in parallel
    for _i_instance in prange(len(X)):
        # prange cast iterator to unit64 with parallel=True
        i_instance = np.int_(_i_instance)
        for i_candidate in range(X[i_instance].shape[1] - query_length + 1):
            # For each candidate subsequence, if it is valid compute distance
            if mask[i_instance][i_candidate]:
                for i_channel in range(n_channels):
                    distance_profiles[i_instance][i_channel, i_candidate] = (
                        numba_distance_function(
                            q[i_channel],
                            X[i_instance][
                                i_channel,
                                i_candidate : i_candidate + query_length,
                            ],
                        )
                    )
    return distance_profiles


@njit(cache=True, fastmath=True, parallel=True)
def _normalized_naive_distance_profile(
    X,
    q,
    mask,
    X_means,
    X_stds,
    q_means,
    q_stds,
    numba_distance_function,
):
    distance_profiles = List()
    query_length = q.shape[1]
    n_channels = q.shape[0]

    # Init distance profile array with unequal length support
    for i_instance in range(len(X)):
        profile_length = X[i_instance].shape[1] - query_length + 1
        distance_profiles.append(np.full((n_channels, profile_length), np.inf))

    # Normalize query once
    q_norm = z_normalize_series_2d_with_mean_std(q, q_means, q_stds)

    # Compute distances in parallel
    for _i_instance in prange(len(X)):
        # prange cast iterator to unit64 with parallel=True
        i_instance = np.int_(_i_instance)
        for i_candidate in range(X[i_instance].shape[1] - query_length + 1):
            # For each candidate subsequence, if it is valid compute distance
            if mask[i_instance][i_candidate]:
                for i_channel in range(n_channels):
                    # Extract and normalize the candidate
                    _C = z_normalize_series_with_mean_std(
                        X[i_instance][
                            i_channel,
                            i_candidate : i_candidate + query_length,
                        ],
                        X_means[i_instance][i_channel, i_candidate],
                        X_stds[i_instance][i_channel, i_candidate],
                    )
                    distance_profiles[i_instance][i_channel, i_candidate] = (
                        numba_distance_function(q_norm[i_channel], _C)
                    )

    return distance_profiles
