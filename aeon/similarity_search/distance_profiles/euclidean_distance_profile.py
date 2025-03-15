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
     Compute the distance profile using the squared Euclidean distance.

    This method calculates the distance profiles between the input time series and
    the query using the squared Euclidean distance. To optimize computation,
    it utilizes a dot product and a rolling sum, reducing redundant calculations.

    Parameters
    ----------
    X : np.ndarray or numba.typed.List
        - If X is a NumPy array, it should have
          the shape ``(n_cases, n_channels, n_timepoints)``.
        - If X contains sequences of unequal lengths, it should be a numba ``TypedList``
          of 2D arrays, each with the shape ``(n_channels, n_timepoints)``.
    q : np.ndarray
        - A 2D array of shape ``(n_channels, query_length)`` representing the query used
          for similarity search.
    mask : np.ndarray
        - A 3D boolean array
          of shape ``(n_cases, n_channels, n_timepoints - query_length + 1)``
          that specifies which parts of the distance profile should be computed.

    Returns
    -------
    distance_profiles : np.ndarray
        - A 3D array of shape ``(n_cases, n_timepoints - query_length + 1)``
          representing the distance profiles between the query ``q`` and
          the input time series ``X``.

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
    Compute a distance profile using brute force.

    It computes the distance profiles between the input time series and the query using
    the specified distance. The search is performed in a brute-force manner without any
    optimizations and can thus be slow.

    Parameters
    ----------
    X : np.ndarray, 3D array of shape ``(n_cases, n_channels, n_timepoints)``
        The input samples. If `X` is an **unequal** length collection, expect a numba
        ``TypedList`` of 2D arrays of shape ``(n_channels, n_timepoints)``.
    q : np.ndarray, 2D array of shape ``(n_channels, query_length)``
        The query used for similarity search.
    mask : np.ndarray,
        3D array of shape ``(n_cases, n_channels, n_timepoints - query_length + 1)``
        Boolean mask indicating for which part of the distance profile the computation
        should be performed.
    X_means : np.ndarray,
        3D array of shape ``(n_cases, n_channels, n_timepoints - query_length + 1)``
        Means of each subsequence of ``X`` of size ``query_length``. Should be a numba
        ``TypedList`` if ``X`` is of unequal length.
    X_stds : np.ndarray,
        3D array of shape ``(n_cases, n_channels, n_timepoints - query_length + 1)``
        Standard deviations of each subsequence of ``X`` of size ``query_length``.
        Should be a numba ``TypedList`` if ``X`` is of unequal length.
    q_means : np.ndarray, 1D array of shape ``(n_channels,)``
        Means of the query ``q``.
    q_stds : np.ndarray, 1D array of shape ``(n_channels,)``
        Standard deviations of the query ``q``.

    Returns
    -------
    distance_profiles : np.ndarray,
        3D array of shape ``(n_cases, n_timepoints - query_length + 1)``
        The computed distance profile between ``q`` and the input time series ``X``.

    """
    distance_profiles = normalised_squared_distance_profile(
        X, q, mask, X_means, X_stds, q_means, q_stds
    )
    # Need loop as we can return a list of np array in the unequal length case
    for i in range(len(distance_profiles)):
        distance_profiles[i] = distance_profiles[i] ** 0.5
    return distance_profiles
