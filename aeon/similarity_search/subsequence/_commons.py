"""Helper functions for subsequence similarity search."""

__maintainer__ = ["baraline"]

import numpy as np
from numba import njit
from scipy.signal import convolve

# Re-export commonly used functions from the parent commons module
from aeon.similarity_search._commons import _inverse_distance_profile  # noqa: F401


@njit(cache=True)
def _extract_top_k_from_dist_profile_one_series(
    dist_profile: np.ndarray,
    k: int,
    dist_threshold: float,
    allow_trivial_matches: bool,
    exclusion_size: int,
) -> tuple:
    """
    Extract top-k matches from a 1D distance profile.

    Finds the k smallest distances in the profile, optionally enforcing
    an exclusion zone around each match to prevent trivial (neighboring)
    matches.

    Parameters
    ----------
    dist_profile : np.ndarray, 1D array of shape (n_candidates,)
        Distance profile from which to extract matches.
    k : int
        Maximum number of matches to return.
    dist_threshold : float
        Maximum allowed distance. Matches with distance > threshold are excluded.
    allow_trivial_matches : bool
        If False, enforce exclusion zones around matches to prevent
        neighboring positions from being returned as separate matches.
    exclusion_size : int
        Size of the exclusion zone around each match. Only used when
        ``allow_trivial_matches=False``.

    Returns
    -------
    top_k_indexes : np.ndarray, 1D array of shape (n_matches,)
        Indexes of the best matches in the distance profile.
    top_k_distances : np.ndarray, 1D array of shape (n_matches,)
        Distances of the best matches.
    """
    n_candidates = len(dist_profile)

    # Initialize output arrays
    top_k_indexes = np.empty(k, dtype=np.int64)
    top_k_distances = np.empty(k, dtype=np.float64)
    n_found = 0

    # Track which positions are excluded
    is_excluded = np.zeros(n_candidates, dtype=np.bool_)

    # Find k best matches
    for _ in range(k):
        best_idx = -1
        best_dist = np.inf

        # Find the minimum non-excluded distance
        for i in range(n_candidates):
            if not is_excluded[i] and dist_profile[i] < best_dist:
                best_dist = dist_profile[i]
                best_idx = i

        # Check if we found a valid match
        if best_idx == -1 or best_dist > dist_threshold:
            break

        # Store the match
        top_k_indexes[n_found] = best_idx
        top_k_distances[n_found] = best_dist
        n_found += 1

        # Apply exclusion zone
        if not allow_trivial_matches:
            lb = max(0, best_idx - exclusion_size)
            ub = min(n_candidates, best_idx + exclusion_size + 1)
            for j in range(lb, ub):
                is_excluded[j] = True
        else:
            is_excluded[best_idx] = True

    return top_k_indexes[:n_found], top_k_distances[:n_found]


def extract_top_k_from_dist_profiles_2d(
    dist_profiles: np.ndarray,
    k: int,
    dist_threshold: float,
    allow_trivial_matches: bool,
    exclusion_size: int,
) -> tuple:
    """
    Extract top-k matches from a 2D array of distance profiles.

    Finds the k smallest distances across all series in a collection,
    returning ``(i_case, i_timestamp)`` pairs indicating where each match
    was found.

    Parameters
    ----------
    dist_profiles : np.ndarray, 2D array of shape (n_cases, n_candidates)
        Distance profiles for each series in the collection.
    k : int
        Maximum number of matches to return.
    dist_threshold : float
        Maximum allowed distance. Matches with distance > threshold are excluded.
    allow_trivial_matches : bool
        If False, enforce exclusion zones around matches within each series
        to prevent neighboring positions from being returned as separate matches.
    exclusion_size : int
        Size of the exclusion zone around each match within a series.
        Only used when ``allow_trivial_matches=False``.

    Returns
    -------
    top_k_indexes : np.ndarray, 2D array of shape (n_matches, 2)
        Indexes of the best matches as ``(i_case, i_timestamp)`` pairs.
    top_k_distances : np.ndarray, 1D array of shape (n_matches,)
        Distances of the best matches.
    """
    n_cases, n_candidates = dist_profiles.shape

    # Initialize output arrays
    top_k_indexes = np.empty((k, 2), dtype=np.int64)
    top_k_distances = np.empty(k, dtype=np.float64)
    n_found = 0

    # Track which positions are excluded in each series
    is_excluded = np.zeros((n_cases, n_candidates), dtype=bool)

    # Find k best matches across all series
    for _ in range(k):
        best_case = -1
        best_idx = -1
        best_dist = np.inf

        # Find the minimum non-excluded distance across all series
        for i_case in range(n_cases):
            for i_ts in range(n_candidates):
                if (
                    not is_excluded[i_case, i_ts]
                    and dist_profiles[i_case, i_ts] < best_dist
                ):
                    best_dist = dist_profiles[i_case, i_ts]
                    best_case = i_case
                    best_idx = i_ts

        # Check if we found a valid match
        if best_case == -1 or best_dist > dist_threshold:
            break

        # Store the match
        top_k_indexes[n_found, 0] = best_case
        top_k_indexes[n_found, 1] = best_idx
        top_k_distances[n_found] = best_dist
        n_found += 1

        # Apply exclusion zone within the same series
        if not allow_trivial_matches:
            lb = max(0, best_idx - exclusion_size)
            ub = min(n_candidates, best_idx + exclusion_size + 1)
            is_excluded[best_case, lb:ub] = True
        else:
            is_excluded[best_case, best_idx] = True

    return top_k_indexes[:n_found], top_k_distances[:n_found]


def fft_sliding_dot_product(X, q):
    """
    Use FFT convolution to calculate the sliding window dot product.

    This function applies the Fast Fourier Transform (FFT) to efficiently compute
    the sliding dot product between the input time series ``X`` and the query ``q``.
    The dot product is computed for each channel individually. The sliding window
    approach ensures that the dot product is calculated for every possible subsequence
    of ``X`` that matches the length of ``q``.

    Parameters
    ----------
    X : array, shape=(n_channels, n_timepoints)
        Input time series.
    q : array, shape=(n_channels, query_length)
        Input query.

    Returns
    -------
    out : np.ndarray, 2D array of shape (n_channels, n_timepoints - query_length + 1)
        Sliding dot product between q and X.
    """
    n_channels, n_timepoints = X.shape
    query_length = q.shape[1]
    out = np.zeros((n_channels, n_timepoints - query_length + 1))
    for i in range(n_channels):
        out[i, :] = convolve(np.flipud(q[i, :]), X[i, :], mode="valid").real
    return out


def get_ith_products(X, T, L, ith):
    """
    Compute dot products between X and the i-th subsequence of size L in T.

    Parameters
    ----------
    X : array, shape = (n_channels, n_timepoints_X)
        Input data.
    T : array, shape =  (n_channels, n_timepoints_T)
        Data containing the query.
    L : int
        Overall query length.
    ith : int
        Query starting index in T.

    Returns
    -------
    np.ndarray, 2D array of shape (n_channels, n_timepoints_X - L + 1)
        Sliding dot product between the i-th subsequence of size L in T and X.
    """
    return fft_sliding_dot_product(X, T[:, ith : ith + L])
