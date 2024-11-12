"""Implementation of stomp for euclidean and squared euclidean distance profile."""

from typing import Optional

__maintainer__ = ["baraline"]


from typing import Union

import numpy as np
from numba import njit
from numba.typed import List

from aeon.similarity_search._commons import (
    extract_top_k_and_threshold_from_distance_profiles_one_series,
    get_ith_products,
    numba_roll_1D_no_warparound,
)
from aeon.similarity_search.distance_profiles.squared_distance_profile import (
    _normalised_squared_dist_profile_one_series,
    _squared_dist_profile_one_series,
)
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD


def stomp_euclidean_matrix_profile(
    X: Union[np.ndarray, List],
    T: np.ndarray,
    L: int,
    mask: np.ndarray,
    k: int = 1,
    threshold: float = np.inf,
    inverse_distance: bool = False,
    exclusion_size: Optional[int] = None,
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
    mask : np.ndarray, 2D array of shape (n_cases, n_timepoints - length + 1)
        Boolean mask of the shape of the distance profiles indicating for which part
        of it the distance should be computed. In this context, it is the mask for the
        first query of size L in T. This mask will be updated during the algorithm.
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
    exclusion_size: Optional[int] = None,
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
    mask : np.ndarray, 2D array of shape (n_cases, n_timepoints - length + 1)
        Boolean mask of the shape of the distance profiles indicating for which part
        of it the distance should be computed. In this context, it is the mask for the
        first query of size L in T. This mask will be updated during the algorithm.
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


def stomp_normalised_euclidean_matrix_profile(
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
    exclusion_size: Optional[int] = None,
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
    mask : np.ndarray, 2D array of shape (n_cases, n_timepoints - length + 1)
        Boolean mask of the shape of the distance profiles indicating for which part
        of it the distance should be computed. In this context, it is the mask for the
        first query of size L in T. This mask will be updated during the algorithm.
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
    MP, IP = stomp_normalised_squared_matrix_profile(
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


def stomp_normalised_squared_matrix_profile(
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
    exclusion_size: Optional[int] = None,
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
    mask : np.ndarray, 2D array of shape (n_cases, n_timepoints - length + 1)
        Boolean mask of the shape of the distance profiles indicating for which part
        of it the distance should be computed. In this context, it is the mask for the
        first query of size L in T. This mask will be updated during the algorithm.
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

    MP, IP = _stomp_normalised(
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


def _stomp_normalised(
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
    """
    Compute the Matrix Profile using the STOMP algorithm with normalised distances.

    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        Length of the subsequences used for the distance computation.
    XdotT : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints - L + 1)
        Precomputed dot products between each time series in X and the query series T.
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
    mask : np.ndarray, 2D array of shape (n_cases, n_timepoints - length + 1)
        Boolean mask of the shape of the distance profiles indicating for which part
        of it the distance should be computed. In this context, it is the mask for the
        first query of size L in T. This mask will be updated during the algorithm.
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

    Returns
    -------
    tuple of np.ndarray
        - MP : array of shape (n_queries,)
          Matrix profile distances for each query subsequence.
        - IP : array of shape (n_queries,)
          Indexes of the top matches for each query subsequence.
    """
    n_queries = T.shape[1] - L + 1
    MP = np.empty(n_queries, dtype=object)
    IP = np.empty(n_queries, dtype=object)
    for i_x in range(len(X)):
        for i in range(n_queries):
            dist_profiles = _normalised_squared_dist_profile_one_series(
                XdotT[i_x],
                X_means[i_x],
                X_stds[i_x],
                T_means[:, i],
                T_stds[:, i],
                L,
                T_stds[:, i] <= AEON_NUMBA_STD_THRESHOLD,
            )
            dist_profiles[~mask[i_x]] = np.inf
            if i + 1 < n_queries:
                XdotT[i_x] = _update_dot_products_one_series(
                    X[i_x], T, XdotT[i_x], L, i + 1
                )

            mask[i_x] = numba_roll_1D_no_warparound(mask[i_x], 1, True)
            (
                top_dists,
                top_indexes,
            ) = extract_top_k_and_threshold_from_distance_profiles_one_series(
                dist_profiles,
                i_x,
                k=k,
                threshold=threshold,
                exclusion_size=exclusion_size,
                inverse_distance=inverse_distance,
            )
            if i_x > 0:
                top_dists, top_indexes = _sort_out_tops(
                    top_dists, MP[i], top_indexes, IP[i], k
                )
                MP[i] = top_dists
                IP[i] = top_indexes
            else:
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
    for i_x in range(len(X)):
        for i in range(n_queries):
            Q = T[:, i : i + L]
            dist_profiles = _squared_dist_profile_one_series(XdotT[i_x], X[i_x], Q)
            dist_profiles[~mask[i_x]] = np.inf
            if i + 1 < n_queries:
                XdotT[i_x] = _update_dot_products_one_series(
                    X[i_x], T, XdotT[i_x], L, i + 1
                )

            mask[i_x] = numba_roll_1D_no_warparound(mask[i_x], 1, True)
            (
                top_dists,
                top_indexes,
            ) = extract_top_k_and_threshold_from_distance_profiles_one_series(
                dist_profiles,
                i_x,
                k=k,
                threshold=threshold,
                exclusion_size=exclusion_size,
                inverse_distance=inverse_distance,
            )
            if i_x > 0:
                top_dists, top_indexes = _sort_out_tops(
                    top_dists, MP[i], top_indexes, IP[i], k
                )
                MP[i] = top_dists
                IP[i] = top_indexes
            else:
                MP[i] = top_dists
                IP[i] = top_indexes

    return MP, IP


def _sort_out_tops(top_dists, prev_top_dists, top_indexes, prev_to_indexes, k):
    """
    Sort and combine top distance results from previous and current computations.

    Parameters
    ----------
    top_dists : np.ndarray
        Array of distances from the current computation. Shape should be (n,).
    prev_top_dists : np.ndarray
        Array of distances from previous computations. Shape should be (n,).
    top_indexes : np.ndarray
        Array of indexes corresponding to the top distances from current computation.
        Shape should be (n,).
    prev_to_indexes : np.ndarray
        Array of indexes corresponding to the top distances from previous computations.
        Shape should be (n,).
    k : int, default=1
        The number of best matches to return during predict for each subsequence.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - A 1D numpy array of sorted distances, of length min(k,
          total number of distances).
        - A 1D numpy array of indexes corresponding to the sorted distances,
          of length min(k, total number of distances).
    """
    all_dists = np.concatenate((prev_top_dists, top_dists))
    all_indexes = np.concatenate((prev_to_indexes, top_indexes))
    if k == np.inf:
        return all_dists, all_indexes
    else:
        idx = np.argsort(all_dists)[:k]
        return all_dists[idx], all_indexes[idx]


@njit(cache=True, fastmath=True)
def _update_dot_products_one_series(
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
    n_candidates = X.shape[1] - L + 1

    for i_ft in range(n_channels):
        # first element of all 0 to n-1 candidates * first element of previous query
        _a1 = X[i_ft, : n_candidates - 1] * T[i_ft, i_query - 1]
        # last element of all 1 to n candidates * last element of current query
        _a2 = X[i_ft, L : L - 1 + n_candidates] * T[i_ft, i_query + L - 1]

        XT_products[i_ft, 1:] = XT_products[i_ft, :-1] - _a1 + _a2

        # Compute first dot product
        XT_products[i_ft, 0] = np.sum(Q[i_ft] * X[i_ft, :L])
    return XT_products
