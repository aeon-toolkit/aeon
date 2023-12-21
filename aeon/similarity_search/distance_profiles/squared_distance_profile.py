"""Optimized distance profile for euclidean distance."""

__author__ = ["baraline"]


import numpy as np
from numba import njit, prange

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
    X: array shape (n_cases, n_channels, series_length)
        The input samples.
    q : np.ndarray shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_instances, series_length - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.

    Returns
    -------
    distance_profile : np.ndarray
        shape (n_cases, n_channels, series_length - query_length + 1)
        The distance profile between q and the input time series X independently
        for each channel.

    """
    QX = np.asarray([fft_sliding_dot_product(X[i], q) for i in range(len(X))])
    return _squared_distance_profile(QX, X, q, mask)


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
    X : array, shape (n_instances, n_channels, series_length)
        The input samples.
    q : array, shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_instances, series_length - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.
    X_means : array, shape (n_instances, n_channels, series_length - query_length + 1)
        Means of each subsequences of X of size query_length
    X_stds : array, shape (n_instances, n_channels, series_length - query_length + 1)
        Stds of each subsequences of X of size query_length
    q_means : array, shape (n_channels)
        Means of the query q
    q_stds : array, shape (n_channels)
        Stds of the query q

    Returns
    -------
    distance_profile : np.ndarray
        shape (n_instances, n_channels, series_length - query_length + 1).
        The distance profile between q and the input time series X independently
        for each channel.

    """
    q_length = X.shape[2] - X_means.shape[2] + 1
    QX = np.asarray([fft_sliding_dot_product(X[i], q) for i in range(len(X))])
    return _normalized_squared_distance_profile(
        QX, mask, X_means, X_stds, q_means, q_stds, q_length
    )


@njit(cache=True, fastmath=True)
def _squared_distance_profile(QX, X, q, mask):
    distance_profile = np.full(QX.shape, np.inf)
    for i_instance in range(len(QX)):
        distance_profile[i_instance][
            :, mask[i_instance]
        ] = _squared_dist_profile_one_series(QX[i_instance], X[i_instance], q)[
            :, mask[i_instance]
        ]
    return distance_profile


@njit(cache=True, fastmath=True)
def _squared_dist_profile_one_series(QT, T, Q):
    n_ft, profile_length = QT.shape
    length = Q.shape[1]
    distance_profile = np.zeros((n_ft, profile_length))
    for k in prange(n_ft):
        distance_profile[k] = -2 * QT[k]
        _sum2 = 0
        for j in prange(length):
            _sum2 += T[k, j] ** 2
            distance_profile[k] += Q[k, j] ** 2
        distance_profile[k, 0] += _sum2
        for i in prange(1, profile_length):
            _sum2 += T[k, i + (length - 1)] ** 2 - T[k, i - 1] ** 2
            distance_profile[k, i] += _sum2

    return distance_profile


@njit(cache=True, fastmath=True)
def _normalized_squared_distance_profile(
    QX, mask, X_means, X_stds, q_means, q_stds, q_length
):
    distance_profile = np.full(QX.shape, np.inf)

    for i_instance in range(len(QX)):
        distance_profile[i_instance][
            :, mask[i_instance]
        ] = _normalized_eucldiean_dist_profile_one_series(
            QX[i_instance],
            X_means[i_instance],
            X_stds[i_instance],
            q_means,
            q_stds,
            q_length,
        )[
            :, mask[i_instance]
        ]
    return distance_profile


@njit(cache=True, fastmath=True)
def _normalized_eucldiean_dist_profile_one_series(
    QT,
    T_means,
    T_stds,
    Q_means,
    Q_stds,
    q_length,
):
    # Compute znormalized squared euclidean distance
    n_ft, profile_length = QT.shape
    distance_profile = np.full((n_ft, profile_length), np.inf)
    Q_is_constant = Q_stds <= AEON_NUMBA_STD_THRESHOLD
    for i in prange(profile_length):
        Sub_is_constant = T_stds[:, i] <= AEON_NUMBA_STD_THRESHOLD
        for k in prange(n_ft):
            # Two Constant case
            if Q_is_constant[k] and Sub_is_constant[k]:
                _val = 0
            else:
                # One Constant case
                if Q_is_constant[k] or Sub_is_constant[k]:
                    _val = q_length
                else:
                    denom = q_length * Q_stds[k] * T_stds[k, i]
                    denom = max(denom, AEON_NUMBA_STD_THRESHOLD**2)

                    p = (QT[k, i] - q_length * (Q_means[k] * T_means[k, i])) / denom
                    p = min(p, 1.0)

                    _val = abs(2 * q_length * (1.0 - p))
            distance_profile[k, i] = _val

    return distance_profile
