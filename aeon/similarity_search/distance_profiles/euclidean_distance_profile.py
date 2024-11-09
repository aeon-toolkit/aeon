"""Optimized distance profile for euclidean distance."""

__maintainer__ = ["baraline"]


from typing import Union

import numpy as np
from numba.typed import List

from aeon.similarity_search.distance_profiles.squared_distance_profile import (
    normalised_squared_distance_profile,
    squared_distance_profile,
)


def euclidean_distance_profile(
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
    X: np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a numba TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    q : np.ndarray, 2D array of shape (n_channels, query_length)
        The query used for similarity search.
    mask : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.

    Returns
    -------
    distance_profiles : np.ndarray
        3D array of shape (n_cases, n_timepoints - query_length + 1)
        The distance profile between q and the input time series X.

    """
    distance_profiles = squared_distance_profile(X, q, mask)
    # Need loop as we can return a list of np array in the unequal length case
    for i in range(len(distance_profiles)):
        distance_profiles[i] = distance_profiles[i] ** 0.5
    return distance_profiles


def normalised_euclidean_distance_profile(
    X: Union[np.ndarray, List],
    q: np.ndarray,
    mask: np.ndarray,
    X_means: Union[np.ndarray, List],
    X_stds: Union[np.ndarray, List],
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
    X: np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a numba TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    q : np.ndarray, 2D array of shape (n_channels, query_length)
        The query used for similarity search.
    mask : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.
    X_means : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Means of each subsequences of X of size query_length. Should be a numba
        TypedList if X is unequal length.
    X_stds : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - query_length + 1)  # noqa: E501
        Stds of each subsequences of X of size query_length. Should be a numba
        TypedList if X is unequal length.
    q_means : np.ndarray, 1D array of shape (n_channels)
        Means of the query q
    q_stds : np.ndarray, 1D array of shape (n_channels)

    Returns
    -------
    distance_profiles : np.ndarray
        3D array of shape (n_cases, n_timepoints - query_length + 1)
        The distance profile between q and the input time series X.

    """
    distance_profiles = normalised_squared_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds
    )
    # Need loop as we can return a list of np array in the unequal length case
    for i in range(len(distance_profiles)):
        distance_profiles[i] = distance_profiles[i] ** 0.5
    return distance_profiles
