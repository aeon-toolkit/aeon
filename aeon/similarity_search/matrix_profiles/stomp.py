"""Implementation of stomp for euclidean and squared euclidean distance profile."""

__maintainer__ = ["baraline"]


from typing import Union

import numpy as np
from numba import njit
from numba.typed import List

from aeon.similarity_search._commons import (
    extract_top_k_and_threshold_from_distance_profiles,
    get_ith_products,
    numba_roll_2D_no_warparound,
)
from aeon.similarity_search.distance_profiles.squared_distance_profile import (
    _normalized_squared_distance_profile,
    _squared_distance_profile,
)


def stomp_euclidean_matrix_profile(
    X: Union[np.ndarray, List],
    T: np.ndarray,
    L: int,
    mask: np.ndarray,
    k: int = 1,
    threshold: float = np.inf,
    inverse_distance: bool = False,
    exclusion_size: int = None,
):
    """
    Compute a euclidean euclidean matrix profile using STOMP [1]_.

    This improves on the naive matrix profile by updating the dot products for each
    sucessive query in T instead of recomputing them.

    Parameters
    ----------
    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        The length of the subsequences considered during the search. This parameter
        cannot be larger than n_timepoints and series_length.
    k : int, default=1
        The number of best matches to return during predict for each subsequence.
    threshold : float, default=np.inf
        The number of best matches to return during predict for each subsequence.
    inverse_distance : bool, default=False
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    exclusion_size : int, optional
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestomp - exclusion_size` and
        :math:`id_timestomp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestomp` was already selected. By default,
        the value None means that this is not used.

    References
    ----------
    .. [1] Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
    Hundred Million Barrier for Time Series Motifs and Joins. Yan Zhu, Zachary
    Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah
    Mueen, Philip Berisk and Eamonn Keogh. IEEE ICDM 2016

    Returns
    -------
    Tuple(ndarray, ndarray)
        The first array, of shape ``(series_length - length + 1, n_matches)``,
        contains the distance between all the queries of size length and their best
        matches in X_. The second array, of shape
        ``(series_length - L + 1, n_matches, 2)``, contains the indexes of these
        matches as ``(id_sample, id_timepoint)``. The corresponding match can be
        retrieved as ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

    """
    MP, IP = stomp_squared_matrix_profile(
        X,
        T,
        L,
        mask,
        k=k,
        threshold=threshold,
        exclusion_size=exclusion_size,
        inverse_distance=inverse_distance,
    )
    for i in range(len(MP)):
        MP[i] = MP[i] ** 0.5
    return MP, IP


def stomp_squared_matrix_profile(
    X: Union[np.ndarray, List],
    T: np.ndarray,
    L: int,
    mask: np.ndarray,
    k: int = 1,
    threshold: float = np.inf,
    inverse_distance: bool = False,
    exclusion_size: int = None,
):
    """
    Compute a squared euclidean matrix profile using STOMP [1]_.

    This improves on the naive matrix profile by updating the dot products for each
    sucessive query in T instead of recomputing them.

    Parameters
    ----------
    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        The length of the subsequences considered during the search. This parameter
        cannot be larger than n_timepoints and series_length.
    k : int, default=1
        The number of best matches to return during predict for each subsequence.
    threshold : float, default=np.inf
        The number of best matches to return during predict for each subsequence.
    inverse_distance : bool, default=False
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    exclusion_size : int, optional
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestomp - exclusion_size` and
        :math:`id_timestomp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestomp` was already selected. By default,
        the value None means that this is not used.

    References
    ----------
    .. [1] Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
    Hundred Million Barrier for Time Series Motifs and Joins. Yan Zhu, Zachary
    Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah
    Mueen, Philip Berisk and Eamonn Keogh. IEEE ICDM 2016

    Returns
    -------
    Tuple(ndarray, ndarray)
        The first array, of shape ``(series_length - length + 1, n_matches)``,
        contains the distance between all the queries of size length and their best
        matches in X_. The second array, of shape
        ``(series_length - L + 1, n_matches, 2)``, contains the indexes of these
        matches as ``(id_sample, id_timepoint)``. The corresponding match can be
        retrieved as ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

    """
    XdotT = [get_ith_products(X[i], T, L, 0) for i in range(len(X))]
    if isinstance(X, np.ndarray):
        XdotT = np.asarray(XdotT)
    elif isinstance(X, List):
        XdotT = List(XdotT)

    MP, IP = _stomp(
        X,
        T,
        XdotT,
        L,
        mask,
        k,
        threshold,
        exclusion_size,
        inverse_distance,
    )
    return MP, IP


def stomp_normalized_euclidean_matrix_profile(
    X: Union[np.ndarray, List],
    T: np.ndarray,
    L: int,
    X_means: Union[np.ndarray, List],
    X_stds: Union[np.ndarray, List],
    T_means: np.ndarray,
    T_stds: np.ndarray,
    mask: np.ndarray,
    k: int = 1,
    threshold: float = np.inf,
    inverse_distance: bool = False,
    exclusion_size: int = None,
):
    """
    Compute a euclidean matrix profile using STOMP [1]_.

    This improves on the naive matrix profile by updating the dot products for each
    sucessive query in T instead of recomputing them.

    Parameters
    ----------
    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        The length of the subsequences considered during the search. This parameter
        cannot be larger than n_timepoints and series_length.
    X_means : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Means of each subsequences of X of size L. Should be a numba TypedList if X is
        unequal length.
    X_stds : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Stds of each subsequences of X of size L. Should be a numba TypedList if X is
        unequal length.
    T_means : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Means of each subsequences of T of size L.
    T_stds : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Stds of each subsequences of T of size L.
    k : int, default=1
        The number of best matches to return during predict for each subsequence.
    threshold : float, default=np.inf
        The number of best matches to return during predict for each subsequence.
    inverse_distance : bool, default=False
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    exclusion_size : int, optional
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestomp - exclusion_size` and
        :math:`id_timestomp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestomp` was already selected. By default,
        the value None means that this is not used.

    References
    ----------
    .. [1] Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
    Hundred Million Barrier for Time Series Motifs and Joins. Yan Zhu, Zachary
    Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah
    Mueen, Philip Berisk and Eamonn Keogh. IEEE ICDM 2016

    Returns
    -------
    Tuple(ndarray, ndarray)
        The first array, of shape ``(series_length - length + 1, n_matches)``,
        contains the distance between all the queries of size length and their best
        matches in X_. The second array, of shape
        ``(series_length - L + 1, n_matches, 2)``, contains the indexes of these
        matches as ``(id_sample, id_timepoint)``. The corresponding match can be
        retrieved as ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

    """
    MP, IP = stomp_normalized_squared_matrix_profile(
        X,
        T,
        L,
        X_means,
        X_stds,
        T_means,
        T_stds,
        mask,
        k=k,
        threshold=threshold,
        exclusion_size=exclusion_size,
        inverse_distance=inverse_distance,
    )
    for i in range(len(MP)):
        MP[i] = MP[i] ** 0.5
    return MP, IP


def stomp_normalized_squared_matrix_profile(
    X: Union[np.ndarray, List],
    T: np.ndarray,
    L: int,
    X_means: Union[np.ndarray, List],
    X_stds: Union[np.ndarray, List],
    T_means: np.ndarray,
    T_stds: np.ndarray,
    mask: np.ndarray,
    k: int = 1,
    threshold: float = np.inf,
    inverse_distance: bool = False,
    exclusion_size: int = None,
):
    """
    Compute a squared euclidean matrix profile using STOMP [1]_.

    This improves on the naive matrix profile by updating the dot products for each
    sucessive query in T instead of recomputing them.

    Parameters
    ----------
    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        The length of the subsequences considered during the search. This parameter
        cannot be larger than n_timepoints and series_length.
    X_means : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Means of each subsequences of X of size L. Should be a numba TypedList if X is
        unequal length.
    X_stds : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Stds of each subsequences of X of size L. Should be a numba TypedList if X is
        unequal length.
    T_means : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Means of each subsequences of T of size L.
    T_stds : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Stds of each subsequences of T of size L.
    k : int, default=1
        The number of best matches to return during predict for each subsequence.
    threshold : float, default=np.inf
        The number of best matches to return during predict for each subsequence.
    inverse_distance : bool, default=False
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    exclusion_size : int, optional
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestomp - exclusion_size` and
        :math:`id_timestomp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestomp` was already selected. By default,
        the value None means that this is not used.

    References
    ----------
    .. [1] Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
    Hundred Million Barrier for Time Series Motifs and Joins. Yan Zhu, Zachary
    Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah
    Mueen, Philip Berisk and Eamonn Keogh. IEEE ICDM 2016

    Returns
    -------
    Tuple(ndarray, ndarray)
        The first array, of shape ``(series_length - length + 1, n_matches)``,
        contains the distance between all the queries of size length and their best
        matches in X_. The second array, of shape
        ``(series_length - L + 1, n_matches, 2)``, contains the indexes of these
        matches as ``(id_sample, id_timepoint)``. The corresponding match can be
        retrieved as ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

    """
    XdotT = [get_ith_products(X[i], T, L, 0) for i in range(len(X))]
    if isinstance(X, np.ndarray):
        XdotT = np.asarray(XdotT)
    elif isinstance(X, List):
        XdotT = List(XdotT)

    MP, IP = _stomp_normalized(
        X,
        T,
        XdotT,
        X_means,
        X_stds,
        T_means,
        T_stds,
        L,
        mask,
        k,
        threshold,
        exclusion_size,
        inverse_distance,
    )
    return MP, IP


def _stomp_normalized(
    X,
    T,
    XdotT,
    X_means,
    X_stds,
    T_means,
    T_stds,
    L,
    mask,
    k,
    threshold,
    exclusion_size,
    inverse_distance,
):
    n_queries = T.shape[1] - L + 1
    MP = np.empty(n_queries, dtype=object)
    IP = np.empty(n_queries, dtype=object)
    for i in range(n_queries):
        dist_profiles, XdotT, mask = _compute_normalized_profile_and_update(
            X, T, XdotT, mask, X_means, X_stds, T_means[i], T_stds[i], L, i
        )
        top_dists, top_indexes = extract_top_k_and_threshold_from_distance_profiles(
            dist_profiles,
            k=k,
            threshold=threshold,
            exclusion_size=exclusion_size,
            inverse_distance=inverse_distance,
        )
        MP[i] = top_dists
        IP[i] = top_indexes
    return MP, IP


def _stomp(
    X,
    T,
    XdotT,
    L,
    mask,
    k,
    threshold,
    exclusion_size,
    inverse_distance,
):
    n_queries = T.shape[1] - L + 1
    MP = np.empty(n_queries, dtype=object)
    IP = np.empty(n_queries, dtype=object)
    for i in range(n_queries):
        dist_profiles, XdotT, mask = _compute_profile_and_update(
            X, T, XdotT, mask, L, i
        )
        if isinstance(X, np.ndarray):
            dist_profiles = np.asarray(dist_profiles).sum(axis=1)
        else:
            dist_profiles = List([dist_profiles[i].sum(axis=0) for i in range(len(X))])

        top_dists, top_indexes = extract_top_k_and_threshold_from_distance_profiles(
            dist_profiles,
            k=k,
            threshold=threshold,
            exclusion_size=exclusion_size,
            inverse_distance=inverse_distance,
        )
        MP[i] = top_dists
        IP[i] = top_indexes

    return MP, IP


@njit(cache=True)
def _compute_profile_and_update(X, T, XdotT, mask, L, i_query):
    Q = T[:, i_query : i_query + L]
    dist_profiles = _squared_distance_profile(XdotT, X, Q, mask)
    if i_query + 1 < T.shape[1] - L + 1:
        XdotT = _update_dot_products(X, T, XdotT, L, i_query + 1)
        mask = numba_roll_2D_no_warparound(mask, 1, True)
    return dist_profiles, XdotT, mask


@njit(cache=True)
def _compute_normalized_profile_and_update(
    X, T, XdotT, mask, X_means, X_stds, T_means, T_stds, L, i_query
):
    dist_profiles = _normalized_squared_distance_profile(
        XdotT, mask, X_means, X_stds, T_means, T_stds, L
    )
    if i_query + 1 < T.shape[1] - L + 1:
        XdotT = _update_dot_products(X, T, XdotT, L, i_query + 1)
        mask = numba_roll_2D_no_warparound(mask, 1, True)
    return dist_profiles, XdotT, mask


@njit(cache=True, fastmath=True)
def _update_dot_products(
    X,
    T,
    XT_products,
    L,
    i_query,
):
    """
    Update dot products of the i-th query of size L in T from the dot products of i-1.

    Parameters
    ----------
    X: np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        Input time series on which the sliding dot product is computed.
    T: np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        The length of the subsequences considered during the search. This parameter
        cannot be larger than n_timepoints and series_length.
    i_query : int
        Query starting index in T.

    Returns
    -------
    XT_products : np.ndarray of shape (n_cases, n_channels, n_timepoints - L + 1)
        Sliding dot product between the i-th subsequence of size L in T and X.

    """
    n_channels = T.shape[0]
    Q = T[:, i_query : i_query + L]
    for i in range(len(X)):
        n_candidates = X[i].shape[1] - L + 1

        # first element of all 0 to n-1 candidates * first element of previous query
        _a1 = X[i][:, : n_candidates - 1] * T[:, i_query - 1][:, np.newaxis]
        # last element of all 1 to n candidates * last element of current query
        _a2 = X[i][:, L : L - 1 + n_candidates] * T[:, i_query + L - 1][:, np.newaxis]

        XT_products[i][:, 1:] = XT_products[i][:, :-1] - _a1 + _a2
        # Compute first dot product
        for i_ft in range(n_channels):
            XT_products[i][i_ft, 0] = np.sum(Q[i_ft] * X[i][i_ft, :L])
    return XT_products
