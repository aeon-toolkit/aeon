"""Optimized distance profile for euclidean distance."""

__maintainer__ = ["baraline"]


import numpy as np
from numba import njit, prange
from numba.typed import List

from aeon.similarity_search.distance_profiles._commons import fft_sliding_dot_product
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD


def squared_distance_profile(X, q, mask):
    """
    Compute a distance profile using the squared Euclidean distance.

    It computes the distance profiles between the input time series and the query using
    the squared Euclidean distance. The distance between the query and a candidate is
    comptued using a dot product and a rolling sum to avoid recomputing parts of the
    operation.

    Parameters
    ----------
    X: array shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    q : np.ndarray shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_cases, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.

    Returns
    -------
    distance_profile : np.ndarray
        shape (n_cases, n_channels, n_timepoints - query_length + 1)
        The distance profile between q and the input time series X independently
        for each channel.

    """
    QX = [fft_sliding_dot_product(X[i], q) for i in range(len(X))]
    if isinstance(X, np.ndarray):
        QX = np.asarray(QX)
    elif isinstance(X, List):
        QX = List(QX)
    distance_profiles = _squared_distance_profile(QX, X, q, mask)
    if isinstance(X, np.ndarray):
        distance_profiles = np.asarray(distance_profiles)
    return distance_profiles


def normalized_squared_distance_profile(
    X,
    q,
    mask,
    X_means,
    X_stds,
    q_means,
    q_stds,
):
    """
    Compute a distance profile in a brute force way.

    It computes the distance profiles between the input time series and the query using
    the specified distance. The search is made in a brute force way without any
    optimizations and can thus be slow.

    Parameters
    ----------
    X : array, shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    q : array, shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_cases, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.
    X_means : array, shape (n_cases, n_channels, n_timepoints - query_length + 1)
        Means of each subsequences of X of size query_length
    X_stds : array, shape (n_cases, n_channels, n_timepoints - query_length + 1)
        Stds of each subsequences of X of size query_length
    q_means : array, shape (n_channels)
        Means of the query q
    q_stds : array, shape (n_channels)
        Stds of the query q

    Returns
    -------
    distance_profiles : np.ndarray
        shape (n_cases, n_channels, n_timepoints - query_length + 1).
        The distance profile between q and the input time series X independently
        for each channel.

    """
    query_length = q.shape[1]
    QX = [fft_sliding_dot_product(X[i], q) for i in range(len(X))]
    if isinstance(X, np.ndarray):
        QX = np.asarray(QX)
    elif isinstance(X, List):
        QX = List(QX)

    distance_profiles = _normalized_squared_distance_profile(
        QX, mask, X_means, X_stds, q_means, q_stds, query_length
    )
    if isinstance(X, np.ndarray):
        distance_profiles = np.asarray(distance_profiles)
    return distance_profiles


@njit(cache=True, fastmath=True, parallel=True)
def _squared_distance_profile(QX, X, q, mask):
    distance_profiles = List()
    query_length = q.shape[1]
    n_channels = q.shape[0]

    # Init distance profile array with unequal length support
    for i_instance in range(len(X)):
        profile_length = X[i_instance].shape[1] - query_length + 1
        distance_profiles.append(np.full((n_channels, profile_length), np.inf))

    for _i_instance in prange(len(QX)):
        # prange cast iterator to unit64 with parallel=True
        i_instance = np.int_(_i_instance)

        distance_profiles[i_instance][:, mask[i_instance]] = (
            _squared_dist_profile_one_series(QX[i_instance], X[i_instance], q)[
                :, mask[i_instance]
            ]
        )
    return distance_profiles


@njit(cache=True, fastmath=True)
def _squared_dist_profile_one_series(QT, T, Q):
    n_channels, profile_length = QT.shape
    query_length = Q.shape[1]
    distance_profile = -2 * QT
    for k in prange(n_channels):
        _sum = 0
        _qsum = 0
        for j in prange(query_length):
            _sum += T[k, j] ** 2
            _qsum += Q[k, j] ** 2

        distance_profile[k, :] += _qsum
        distance_profile[k, 0] += _sum
        for i in prange(1, profile_length):
            _sum += T[k, i + (query_length - 1)] ** 2 - T[k, i - 1] ** 2
            distance_profile[k, i] += _sum
    return distance_profile


@njit(cache=True, fastmath=True, parallel=True)
def _normalized_squared_distance_profile(
    QX, mask, X_means, X_stds, q_means, q_stds, query_length
):
    distance_profiles = List()
    n_channels = q_means.shape[0]
    Q_is_constant = q_stds <= AEON_NUMBA_STD_THRESHOLD
    # Init distance profile array with unequal length support
    for i_instance in range(len(QX)):
        profile_length = QX[i_instance].shape[1]
        distance_profiles.append(np.full((n_channels, profile_length), np.inf))

    for _i_instance in prange(len(QX)):
        # prange cast iterator to unit64 with parallel=True
        i_instance = np.int_(_i_instance)

        distance_profiles[i_instance][:, mask[i_instance]] = (
            _normalized_eucldiean_dist_profile_one_series(
                QX[i_instance],
                X_means[i_instance],
                X_stds[i_instance],
                q_means,
                q_stds,
                query_length,
                Q_is_constant,
            )[:, mask[i_instance]]
        )
    return distance_profiles


@njit(cache=True, fastmath=True)
def _normalized_eucldiean_dist_profile_one_series(
    QT, T_means, T_stds, Q_means, Q_stds, query_length, Q_is_constant
):
    # Compute znormalized squared euclidean distance
    n_channels, profile_length = QT.shape
    distance_profile = np.full((n_channels, profile_length), np.inf)

    for i in prange(profile_length):
        Sub_is_constant = T_stds[:, i] <= AEON_NUMBA_STD_THRESHOLD
        for k in prange(n_channels):
            # Two Constant case
            if Q_is_constant[k] and Sub_is_constant[k]:
                _val = 0
            # One Constant case
            elif Q_is_constant[k] or Sub_is_constant[k]:
                _val = query_length
            else:
                denom = query_length * Q_stds[k] * T_stds[k, i]

                p = (QT[k, i] - query_length * (Q_means[k] * T_means[k, i])) / denom
                p = min(p, 1.0)

                _val = abs(2 * query_length * (1.0 - p))
            distance_profile[k, i] = _val

    return distance_profile
