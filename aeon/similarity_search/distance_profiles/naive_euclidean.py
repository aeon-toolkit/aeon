    r"""
    Compute a euclidean distance profile in a brute force way.
    
    This function computes the distance profiles between the input time series and the query using
    the Euclidean distance. The search is made in a brute force way without any
    optimizations and can thus be slow.
    
    Parameters
    ----------
    X: array shape (n_cases, n_channels, series_length)
    The input samples.
    q : np.ndarray shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_instances, n_channels, n_timestamps - (q_length - 1))
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.
    
    Returns
    -------
    distance_profile : np.ndarray
        shape (n_cases, n_channels, series_length - query_length + 1)
        The distance profile between q and the input time series X independently
        for each channel.
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 1, 100)
    >>> q = np.random.rand(1, 50)
    >>> mask = np.ones((10, 1, 51), dtype=bool)
    >>> distance_profile = naive_euclidean_profile(X, q, mask)
    >>> print(distance_profile.shape)
    (10, 1, 51)

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
    q : np.ndarray shape (n_channels, query_length)
        The query used for similarity search.
    mask : array, shape (n_instances, n_channels, n_timestamps - (q_length - 1))
        Boolean mask of the shape of the distance profile indicating for which part
        of it the distance should be computed.

    Returns
    -------
    distance_profile : np.ndarray
        shape (n_cases, n_channels, series_length - query_length + 1)
        The distance profile between q and the input time series X independently
        for each channel.

    """
    return _naive_euclidean_profile(X, q, mask)


@njit(cache=True, fastmath=True)
def _naive_euclidean_profile(X, q, mask):
    n_instances, n_channels, X_length, q_length, profile_size = _get_input_sizes(X, q)
    distance_profile = np.full((n_instances, n_channels, profile_size), np.inf)

    for i_instance in range(n_instances):
        for i_channel in range(n_channels):
            for i_candidate in range(profile_size):
                if mask[i_instance, i_channel, i_candidate]:
                    distance_profile[
                        i_instance, i_channel, i_candidate
                    ] = euclidean_distance(
                        q[i_channel],
                        X[i_instance, i_channel, i_candidate : i_candidate + q_length],
                    )

    return distance_profile
