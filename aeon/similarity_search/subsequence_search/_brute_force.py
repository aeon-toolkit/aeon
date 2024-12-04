"""Implementation of matrix profile with brute force."""

from typing import Optional

__maintainer__ = ["baraline"]


import numpy as np
from numba import njit, prange
from numba.typed import List

from aeon.similarity_search.subsequence_search._commons import (
    _extract_top_k_from_dist_profile,
    _inverse_distance_profile_list,
)
from aeon.similarity_search.subsequence_search.base import BaseMatrixProfile
from aeon.utils.numba.general import (
    get_all_subsequences,
    z_normalise_series_3d,
    z_normalize_series_2d,
)

# TODO : check function params and make docstrings
# TODO : make tests


class BruteForceMatrixProfile(BaseMatrixProfile):
    """."""

    def compute_matrix_profile(
        self,
        k,
        threshold,
        exclusion_size,
        inverse_distance,
        allow_overlap,
        X: Optional[np.ndarray] = None,
        X_index: Optional[int] = None,
    ):
        """
        .

        Parameters
        ----------
        k : TYPE
            DESCRIPTION.
        threshold : TYPE
            DESCRIPTION.
        exclusion_size : TYPE
            DESCRIPTION.
        inverse_distance : TYPE
            DESCRIPTION.
        X : Optional[np.ndarray], optional
            DESCRIPTION. The default is None.
        X_index : Optional[int], optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        MP : TYPE
            DESCRIPTION.
        IP : TYPE
            DESCRIPTION.

        """
        # pairwise if none
        if X is None:
            MP = []
            IP = []
            for i in range(len(self.X_)):
                _MP, _IP = self.compute_matrix_profile(
                    k,
                    threshold,
                    exclusion_size,
                    inverse_distance,
                    X=self.X_[i],
                    X_index=i,
                )
                MP.append(_MP)
                IP.append(_IP)
        else:
            MP, IP = _naive_squared_matrix_profile(
                self.X_,
                X,
                self.length,
                X_index,
                k,
                allow_overlap,
                threshold,
                exclusion_size,
                inverse_distance,
                normalize=self.normalize,
            )

        return MP, IP

    def compute_distance_profile(self, X: np.ndarray):
        """
        Compute the distance profile of X to all samples in X_.

        Parameters
        ----------
        X : np.ndarray, 2D array of shape (n_channels, length)
            The query to use to compute the distance profiles.

        Returns
        -------
        distance_profiles : np.ndarray, 2D array of shape (n_cases, n_candidates)
            The distance profile of X to all samples in X_. The ``n_candidates`` value
            is equal to ``n_timepoins - length + 1``. If X_ is an unequal length
            collection, returns a numba typed list instead of an ndarray.

        """
        distance_profiles = _naive_squared_distance_profile(
            self.X_, X, normalize=self.normalize
        )

        if not self.metadata_["unequal_length"]:
            distance_profiles = np.asarray(distance_profiles)
        return distance_profiles


@njit(cache=True, fastmath=True)
def _compute_dist_profile(X_subs, q):
    """
    Compute the distance profile between subsequences and a query.

    Parameters
    ----------
    X_subs : array, shape=(n_samples, n_channels, query_length)
        Input subsequences extracted from a time series.
    q : array, shape=(n_channels, query_length)
        Query used for the distance computation

    Returns
    -------
    dist_profile : np.ndarray, 1D array of shape (n_samples)
        The distance between the query all subsequences.

    """
    n_candidates, n_channels, q_length = X_subs.shape
    dist_profile = np.zeros(n_candidates)
    for i in range(n_candidates):
        for j in range(n_channels):
            for k in range(q_length):
                dist_profile[i] += (X_subs[i, j, k] - q[j, k]) ** 2
    return dist_profile


@njit(cache=True, fastmath=True, parallel=True)
def _naive_squared_distance_profile(
    X,
    Q,
    normalize=False,
):
    """
    Compute a squared euclidean distance profile.

    Parameters
    ----------
    X : array, shape=(n_samples, n_channels, n_timepoints)
        Input time series dataset to search in.
    Q : array, shape=(n_channels, query_length)
        Query used during the search.
    normalize : bool
        Wheter to use a z-normalized distance.

    Returns
    -------
    out : np.ndarray, 1D array of shape (n_samples, n_timepoints_t - query_length + 1)
        The distance between the query and all candidates in X.

    """
    query_length = Q.shape[1]
    dist_profiles = List()
    # Init distance profile array with unequal length support
    for i in range(len(X)):
        dist_profiles.append(np.zeros(X[i].shape[1] - query_length + 1))
    if normalize:
        Q = z_normalize_series_2d(Q)
    else:
        Q = Q.astype(np.float64)

    for i in prange(len(X)):
        X_subs = get_all_subsequences(X[i], query_length, 1)
        if normalize:
            X_subs = z_normalise_series_3d(X_subs)

        dist_profile = _compute_dist_profile(X_subs, Q)
        dist_profiles[i] = dist_profile
    return dist_profiles


@njit(cache=True, fastmath=True, parallel=True)
def _naive_squared_matrix_profile(
    X,
    T,
    L,
    k,
    T_index,
    threshold,
    inverse_distance,
    allow_overlap,
    exclusion_size,
    normalize=False,
):
    """
    Compute a squared euclidean matrix profile.

    Parameters
    ----------
    X : array, shape=(n_samples, n_channels, n_timepoints_x)
        Input time series dataset to search in.
    T : array, shape=(n_channels, n_timepoints_t)
        Time series from which queries are extracted.
    query_length : int
        Length of the queries to extract from T.
    T_index : int,
        If ``X`` is a subsequence of the database given in fit, specify its starting
        index as (i_case, i_timestamp). If specified, this subsequence and the
        neighboring ones (according to ``exclusion_factor``) won't be considered as
        admissible candidates.
    normalize : bool
        Wheter to use a z-normalized distance.

    Returns
    -------
    out : np.ndarray, 1D array of shape (n_timepoints_t - query_length + 1)
        The minimum distance between each query in T and all candidates in X.
    """
    n_queries = T.shape[1] - L + 1
    MP = List()
    IP = List()

    # Init List to allow parallel, we'll re-use it for all dist profiles
    dist_profiles = List()
    for i_x in range(len(X)):
        dist_profiles.append(np.zeros(X[i_x].shape[1] - L + 1))

    X_subs = List()
    for i in range(len(X)):
        i_subs = get_all_subsequences(X[i], L, 1)
        if normalize:
            i_subs = z_normalise_series_3d(X_subs)
        X_subs.append(i_subs)

    for i_q in range(n_queries):
        Q = T[:, i : i + L]
        if normalize:
            Q = z_normalize_series_2d(Q)
        for i_x in prange(len(X)):
            dist_profiles[i_x][0 : X[i_x].shape[1] - L + 1] = _compute_dist_profile(
                X_subs[i_x], Q
            )

        if T_index is not None:
            _max_timestamp = X[T_index].shape[1] - L
            ub = min(i_q + exclusion_size, _max_timestamp)
            lb = max(0, i_q - exclusion_size)
            dist_profiles[T_index][lb:ub] = np.inf

        if inverse_distance:
            dist_profiles = _inverse_distance_profile_list(dist_profiles)

        # Deal with self-matches
        if T_index is not None:
            _max_timestamp = X[T_index].shape[1] - L
            ub = min(i_q + exclusion_size, _max_timestamp)
            lb = max(0, i_q - exclusion_size)
            dist_profiles[T_index][lb:ub] = np.inf

        top_dists, top_indexes = _extract_top_k_from_dist_profile(
            dist_profiles,
            k,
            threshold,
            allow_overlap,
            exclusion_size,
        )

        MP.append(top_dists)
        IP.append(top_indexes)
    return MP, IP
