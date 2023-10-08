"""Naive Euclidean distance profile."""

__author__ = ["baraline"]

import numpy as np
from numba import njit

from aeon.distances import euclidean_distance
from aeon.similarity_search.distance_profiles._commons import _get_input_sizes


def naive_euclidean_profile(X, Q):
    r"""
    Compute a euclidean distance profile in a brute force way.

    It computes the distance profiles between the input time series and the query using
    the Euclidean distance. The search is made in a brute force way without any
    optimizations and can thus be slow.

    A distance profile between a (univariate) time series :math:`X_i = {x_1, ..., x_m}`
    and a query :math:`Q = {q_1, ..., q_m}` is defined as a vector of size :math:`m-(
    l-1)`, such as :math:`P(X_i, Q) = {d(C_1, Q), ..., d(C_m-(l-1), Q)}` with d the
    Euclidean distance, and :math:`C_j = {x_j, ..., x_{j+(l-1)}}` the j-th candidate
    subsequence of size :math:`l` in :math:`X_i`.

    Parameters
    ----------
    X: array shape (n_cases, n_channels, series_length)
        The input samples.

    Q : np.ndarray shape (n_channels, query_length)
        The query used for similarity search.

    Returns
    -------
    distance_profile : np.ndarray shape (n_cases, series_length - query_length + 1)
        The distance profile between Q and the input time series X.

    """
    return _naive_euclidean_profile(X, Q)


@njit(cache=True, fastmath=True)
def _naive_euclidean_profile(X, Q):
    n_samples, n_channels, X_length, Q_length, search_space_size = _get_input_sizes(
        X, Q
    )
    distance_profile = np.full((n_samples, search_space_size), np.inf)

    # Compute euclidean distance for all candidate in a "brute force" way
    for i_sample in range(n_samples):
        for i_candidate in range(search_space_size):
            distance_profile[i_sample, i_candidate] = euclidean_distance(
                Q, X[i_sample, :, i_candidate : i_candidate + Q_length]
            )

    return distance_profile
