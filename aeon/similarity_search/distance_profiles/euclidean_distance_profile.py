"""Optimized distance profile for euclidean distance."""

__maintainer__ = []


from aeon.similarity_search.distance_profiles.squared_distance_profile import (
    normalized_squared_distance_profile,
    squared_distance_profile,
)


def euclidean_distance_profile(X, q, mask):
    """
    Compute a distance profile using the squared Euclidean distance.

    It computes the distance profiles between the input time series and the query using
    the squared Euclidean distance. The distance between the query and a candidate is
    comptued using a dot product and a rolling sum to avoid recomputing parts of the
    operation.

    Parameters
    ----------
    X: array shape (n_cases, n_channels, n_timepoints)
        The input samples.
    q : np.ndarray shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_cases, n_channels, n_timepoints - query_length + 1)
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.

    Returns
    -------
    distance_profile : np.ndarray
        shape (n_cases, n_channels, n_timepoints - query_length + 1)
        The distance profile between q and the input time series X independently
        for each channel.

    """
    return squared_distance_profile(X, q, mask) ** 0.5


def normalized_euclidean_distance_profile(
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
        The input samples.
    q : array, shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_cases, n_channels, n_timepoints - query_length + 1)
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
    distance_profile : np.ndarray
        shape (n_cases, n_channels, n_timepoints - query_length + 1).
        The distance profile between q and the input time series X independently
        for each channel.

    """
    return (
        normalized_squared_distance_profile(
            X, q, mask, X_means, X_stds, q_means, q_stds
        )
        ** 0.5
    )
