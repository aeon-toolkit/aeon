"""Optimized distance profile for euclidean distance."""

__maintainer__ = ["baraline"]


from typing import Union

import numpy as np
from numba import njit, prange
from numba.typed import List

from aeon.similarity_search._commons import fft_sliding_dot_product
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD


def squared_distance_profile(
    X: Union[np.ndarray, List], q: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Compute a distance profile using the squared Euclidean distance.

    It computes the distance profiles between the input time series and the query using
    the squared Euclidean distance. The distance between the query and a candidate is
    comptued using a dot product and a rolling sum to avoid recomputing parts of the
    operation.

    Parameters
    ----------
    X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a numba TypedList
        2D array of shape (n_channels, n_timepoints)
    q : np.ndarray, 2D array of shape (n_channels, query_length)
        The query used for similarity search.
    mask : np.ndarray, 3D array of shape (n_cases, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.

    Returns
    -------
    distance_profile : np.ndarray
        3D array of shape (n_cases, n_timepoints - query_length + 1)
        The distance profile between q and the input time series X.

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


def normalised_squared_distance_profile(
    X: Union[np.ndarray, List],
    q: np.ndarray,
    mask: np.ndarray,
    X_means: np.ndarray,
    X_stds: np.ndarray,
    q_means: np.ndarray,
    q_stds: np.ndarray,
) -> np.ndarray:
    """
    Compute a distance profile in a brute force way.

    It computes the distance profiles between the input time series and the query using
    the specified distance. The search is made in a brute force way without any
    optimizations and can thus be slow.

    Parameters
    ----------
    X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a numba TypedList
        2D array of shape (n_channels, n_timepoints)
    q : np.ndarray, 2D array of shape (n_channels, query_length)
        The query used for similarity search.
    mask : np.ndarray, 3D array of shape (n_cases, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.
    X_means : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Means of each subsequences of X of size query_length
    X_stds : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Stds of each subsequences of X of size query_length
    q_means : np.ndarray, 1D array of shape (n_channels)
        Means of the query q
    q_stds : np.ndarray, 1D array of shape (n_channels)
        Stds of the query q

    Returns
    -------
    distance_profiles : np.ndarray
        3D array of shape (n_cases, n_timepoints - query_length + 1)
        The distance profile between q and the input time series X.

    """
    query_length = q.shape[1]
    QX = [fft_sliding_dot_product(X[i], q) for i in range(len(X))]
    if isinstance(X, np.ndarray):
        QX = np.asarray(QX)
    elif isinstance(X, List):
        QX = List(QX)

    distance_profiles = _normalised_squared_distance_profile(
        QX, mask, X_means, X_stds, q_means, q_stds, query_length
    )
    if isinstance(X, np.ndarray):
        distance_profiles = np.asarray(distance_profiles)
    return distance_profiles


@njit(cache=True, fastmath=True, parallel=True)
def _squared_distance_profile(QX, X, q, mask):
    """
    Compute squared distance profiles between query subsequence and time series.

    Parameters
    ----------
    QX : List of np.ndarray
        List of precomputed dot products between queries and time series, with each
        element corresponding to a different time series.
        Shape of each array is (n_channels, n_timepoints - query_length + 1).
    X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a numba TypedList
        2D array of shape (n_channels, n_timepoints)
    q : np.ndarray, 2D array of shape (n_channels, query_length)
        The query used for similarity search.
    mask : np.ndarray, 3D array of shape (n_cases, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.

    Returns
    -------
    distance_profiles : np.ndarray
        3D array of shape (n_cases, n_timepoints - query_length + 1)
        The distance profile between q and the input time series X.

    """
    distance_profiles = List()
    query_length = q.shape[1]

    # Init distance profile array with unequal length support
    for i_instance in range(len(X)):
        profile_length = X[i_instance].shape[1] - query_length + 1
        distance_profiles.append(np.full((profile_length), np.inf))

    for _i_instance in prange(len(QX)):
        # prange cast iterator to unit64 with parallel=True
        i_instance = np.int_(_i_instance)

        distance_profiles[i_instance][mask[i_instance]] = (
            _squared_dist_profile_one_series(QX[i_instance], X[i_instance], q)[
                mask[i_instance]
            ]
        )
    return distance_profiles


@njit(cache=True, fastmath=True)
def _squared_dist_profile_one_series(QT, T, Q):
    """
    Compute squared distance profile between query subsequence and a single time series.

    This function calculates the squared distance profile for a single time series by
    leveraging the dot product of the query and time series as well as precomputed sums
    of squares to efficiently compute the squared distances.

    Parameters
    ----------
    QT : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        The dot product between the query and the time series.
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    Q : np.ndarray
        2D array of shape (n_channels, query_length) representing query subsequence.

    Returns
    -------
    distance_profile : np.ndarray
        2D array of shape (n_channels, n_timepoints - query_length + 1)
        The squared distance profile between the query and the input time series.
    """
    n_channels, profile_length = QT.shape
    query_length = Q.shape[1]
    _QT = -2 * QT
    distance_profile = np.zeros(profile_length)
    for k in prange(n_channels):
        _sum = 0
        _qsum = 0
        for j in prange(query_length):
            _sum += T[k, j] ** 2
            _qsum += Q[k, j] ** 2

        distance_profile += _qsum + _QT[k]
        distance_profile[0] += _sum
        for i in prange(1, profile_length):
            _sum += T[k, i + (query_length - 1)] ** 2 - T[k, i - 1] ** 2
            distance_profile[i] += _sum
    return distance_profile


@njit(cache=True, fastmath=True, parallel=True)
def _normalised_squared_distance_profile(
    QX, mask, X_means, X_stds, q_means, q_stds, query_length
):
    """
    Compute the normalised squared distance profiles between query subsequence and input time series.

    Parameters
    ----------
    QX : List of np.ndarray
        List of precomputed dot products between queries and time series, with each element
        corresponding to a different time series.
        Shape of each array is (n_channels, n_timepoints - query_length + 1).
    mask : np.ndarray, 3D array of shape (n_cases, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.
    X_means : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Means of each subsequences of X of size query_length
    X_stds : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Stds of each subsequences of X of size query_length
    q_means : np.ndarray, 1D array of shape (n_channels)
        Means of the query q
    q_stds : np.ndarray, 1D array of shape (n_channels)
        Stds of the query q
    query_length : int
        The length of the query subsequence used for the distance profile computation.

    Returns
    -------
    List of np.ndarray
        List of 2D arrays, each of shape (n_channels, n_timepoints - query_length + 1).
        Each array contains the normalised squared distance profile between the query subsequence and the corresponding time series.
        Entries in the array are set to infinity where the mask is False.
    """
    distance_profiles = List()
    Q_is_constant = q_stds <= AEON_NUMBA_STD_THRESHOLD
    # Init distance profile array with unequal length support
    for i_instance in range(len(QX)):
        profile_length = QX[i_instance].shape[1]
        distance_profiles.append(np.full((profile_length), np.inf))

    for _i_instance in prange(len(QX)):
        # prange cast iterator to unit64 with parallel=True
        i_instance = np.int_(_i_instance)

        distance_profiles[i_instance][mask[i_instance]] = (
            _normalised_squared_dist_profile_one_series(
                QX[i_instance],
                X_means[i_instance],
                X_stds[i_instance],
                q_means,
                q_stds,
                query_length,
                Q_is_constant,
            )[mask[i_instance]]
        )
    return distance_profiles


@njit(cache=True, fastmath=True)
def _normalised_squared_dist_profile_one_series(
    QT, T_means, T_stds, Q_means, Q_stds, query_length, Q_is_constant
):
    """
    Compute the z-normalised squared Euclidean distance profile for one time series.

    Parameters
    ----------
    QT : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        The dot product between the query and the time series.
    T_means : np.ndarray, 1D array of length n_channels
        The mean values of the time series for each channel.

    T_stds : np.ndarray, 2D array of shape (n_channels, profile_length)
        The standard deviations of the time series for each channel and position.
    Q_means : np.ndarray, 1D array of shape (n_channels)
        Means of the query q
    Q_stds : np.ndarray, 1D array of shape (n_channels)
        Stds of the query q
    query_length : int
        The length of the query subsequence used for the distance profile computation.
    Q_is_constant : np.ndarray
        1D array of shape (n_channels,) where each element is a Boolean indicating
        whether the query standard deviation for that channel is less than or equal
        to a specified threshold.

    Returns
    -------
    np.ndarray
        2D array of shape (n_channels, n_timepoints - query_length + 1) containing the
        z-normalised squared distance profile between the query subsequence and the time
        series. Entries are computed based on the z-normalised values, with special
        handling for constant values.
    """
    n_channels, profile_length = QT.shape
    distance_profile = np.zeros(profile_length)

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
            distance_profile[i] += _val

    return distance_profile
