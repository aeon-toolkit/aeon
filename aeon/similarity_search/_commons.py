"""Helper and common function for similarity search estimators and functions."""

__maintainer__ = ["baraline"]

import warnings

import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.signal import convolve

from aeon.utils.numba.general import (
    get_all_subsequences,
    normalise_subsequences,
    sliding_mean_std_one_series,
    z_normalise_series_2d,
)


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


@njit(cache=True, fastmath=True)
def naive_squared_distance_profile(
    X,
    q,
    mask,
    normalise=False,
    X_means=None,
    X_stds=None,
):
    """
    Compute a squared euclidean distance profile.

    Parameters
    ----------
    X : array, shape=(n_samples, n_channels, n_timepoints)
        Input time series dataset to search in.
    q : array, shape=(n_channels, query_length)
        Query used during the search.
    mask : array, shape=(n_samples,  n_timepoints - query_length + 1)
        Boolean mask indicating candidates for which the distance
        profiles computed for each query should be set to infinity.
    normalise : bool
        Wheter to use a z-normalised distance.
    X_means : array, shape=(n_samples, n_channels, n_timepoints - query_length + 1)
        Mean of each candidate (subsequence) of length query_length in X. The
        default is None, meaning that these values will be computed if normalise
        is True. If provided, the computations will be skipped.
    X_stds : array, shape=(n_samples, n_channels, n_timepoints - query_length + 1)
        Standard deviation of each candidate (subsequence) of length query_length
        in X. The default is None, meaning that these values will be computed if
        normalise is True. If provided, the computations will be skipped.

    Returns
    -------
    out : np.ndarray, 1D array of shape (n_samples, n_timepoints_t - query_length + 1)
        The distance between the query and all candidates in X.

    """
    query_length = q.shape[1]
    dist_profiles = List()
    # Init distance profile array with unequal length support
    for i in range(len(X)):
        dist_profiles.append(np.zeros(X[i].shape[1] - query_length + 1))
    if normalise:
        q = z_normalise_series_2d(q)
    else:
        q = q.astype(np.float64)
    for i in range(len(X)):
        # Numba don't support strides with integers ?

        X_subs = get_all_subsequences(X[i].astype(np.float64), query_length, 1)
        if normalise:
            if X_means is None and X_stds is None:
                _X_means, _X_stds = sliding_mean_std_one_series(X[i], query_length, 1)
            else:
                _X_means, _X_stds = X_means[i], X_stds[i]
            X_subs = normalise_subsequences(X_subs, _X_means, _X_stds)
        dist_profile = _compute_dist_profile(X_subs, q)
        dist_profile[~mask[i]] = np.inf
        dist_profiles[i] = dist_profile
    return dist_profiles


@njit(cache=True, fastmath=True)
def naive_squared_matrix_profile(X, T, query_length, mask, normalise=False):
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
    mask : array, shape=(n_samples, n_timepoints_x - query_length + 1)
        Boolean mask indicating candidates for which the distance
        profiles computed for each query should be set to infinity.
    normalise : bool
        Wheter to use a z-normalised distance.

    Returns
    -------
    out : np.ndarray, 1D array of shape (n_timepoints_t - query_length + 1)
        The minimum distance between each query in T and all candidates in X.
    """
    X_subs = List()
    for i in range(len(X)):
        i_subs = get_all_subsequences(X[i].astype(np.float64), query_length, 1)
        if normalise:
            X_means, X_stds = sliding_mean_std_one_series(X[i], query_length, 1)
            i_subs = normalise_subsequences(i_subs, X_means, X_stds)
        X_subs.append(i_subs)

    n_candidates = T.shape[1] - query_length + 1
    mp = np.full(n_candidates, np.inf)

    for i in range(n_candidates):
        q = T[:, i : i + query_length]
        if normalise:
            q = z_normalise_series_2d(q)
        for id_sample in range(len(X)):
            dist_profile = _compute_dist_profile(X_subs[id_sample], q)
            dist_profile[~mask[id_sample]] = np.inf
            mp[i] = min(mp[i], dist_profile.min())
    return mp


def fft_sliding_dot_product(X, q):
    """
    Use FFT convolution to calculate the sliding window dot product.

    This function applies the Fast Fourier Transform (FFT) to efficiently compute
    the sliding dot product between the input time series `X` and the query `q`.
    The dot product is computed for each channel individually. The sliding window
    approach ensures that the dot product is calculated for every possible subsequence
    of `X` that matches the length of `q`

    Parameters
    ----------
    X : array, shape=(n_channels, n_timepoints)
        Input time series
    q : array, shape=(n_channels, query_length)
        Input query

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


@njit(cache=True)
def numba_roll_1D_no_warparound(array, shift, warparound_value):
    """
    Roll the rows of an array.

    Wheter to allow values at the end of the array to appear at the start after
    being rolled out of the array length.

    Parameters
    ----------
    array : np.ndarray of shape (n_columns)
        Array to roll.
    shift : int
        The amount of indexes the values will be rolled on each row of the array.
        Must be inferior or equal to n_columns.
    warparound_value : any type
        A value of the type of array to insert instead of the value that got rolled
        over the array length

    Returns
    -------
    rolled_array : np.ndarray of shape (n_rows, n_columns)
        The rolled array. Can also be a TypedList in the case where n_columns changes
        between rows.

    """
    length = array.shape[0]
    _a1 = array[: length - shift]
    array[shift:] = _a1
    array[:shift] = warparound_value
    return array


@njit(cache=True)
def numba_roll_2D_no_warparound(array, shift, warparound_value):
    """
    Roll the rows of an array.

    Wheter to allow values at the end of the array to appear at the start after
    being rolled out of the array length.

    Parameters
    ----------
    array : np.ndarray of shape (n_rows, n_columns)
        Array to roll. Can also be a TypedList in the case where n_columns changes
        between rows.
    shift : int
        The amount of indexes the values will be rolled on each row of the array.
        Must be inferior or equal to n_columns.
    warparound_value : any type
        A value of the type of array to insert instead of the value that got rolled
        over the array length

    Returns
    -------
    rolled_array : np.ndarray of shape (n_rows, n_columns)
        The rolled array. Can also be a TypedList in the case where n_columns changes
        between rows.

    """
    for i in prange(len(array)):
        length = len(array[i])
        _a1 = array[i][: length - shift]
        array[i][shift:] = _a1
        array[i][:shift] = warparound_value
    return array


@njit(cache=True)
def extract_top_k_and_threshold_from_distance_profiles_one_series(
    distance_profiles,
    id_x,
    k=1,
    threshold=np.inf,
    exclusion_size=None,
    inverse_distance=False,
):
    """
    Extract the top-k smallest values from distance profiles and apply threshold.

    This function processes a distance profile and extracts the top-k smallest
    distance values, optionally applying a threshold to exclude distances above
    a given value. It also optionally handles exclusion zones to avoid selecting
    neighboring timestamps.

    Parameters
    ----------
    distance_profiles : np.ndarray, 2D array of shape (n_cases, n_candidates)
        Precomputed distance profile. Can be a TypedList if n_candidates vary between
        cases.
    id_x : int
        Identifier of the series or subsequence from which the distance profile
        is computed.
    k : int
        Number of matches to returns
    threshold : float
        All matches below this threshold will be returned
    exclusion_size : int or None, optional, default=None
        Size of the exclusion zone around the current subsequence. This prevents
        selecting neighboring subsequences within the specified range, useful for
        avoiding trivial matches in time series data. If set to `None`, no
        exclusion zone is applied.
    inverse_distance : bool, optional
        Wheter to return the worst matches instead of the bests. The default is False.

    Returns
    -------
    top_k_dist : np.ndarray
        Array of the top-k smallest distance values, potentially excluding values above
        the threshold or those within the exclusion zone.
    top_k : np.ndarray
        Array of shape (k, 2) where each row contains the `id_x` identifier and the
        index of the corresponding subsequence (or timestamp) with the top-k smallest
        distances.
    """
    if inverse_distance:
        # To avoid div by 0 case
        distance_profiles += 1e-8
        distance_profiles[distance_profiles != np.inf] = (
            1 / distance_profiles[distance_profiles != np.inf]
        )

    if threshold != np.inf:
        distance_profiles[distance_profiles > threshold] = np.inf

    _argsort = np.argsort(distance_profiles)

    if distance_profiles[distance_profiles <= threshold].shape[0] < k:
        _k = distance_profiles[distance_profiles <= threshold].shape[0]
    elif _argsort.shape[0] < k:
        _k = _argsort.shape[0]
    else:
        _k = k

    if exclusion_size is None:
        indexes = np.zeros((_k, 2), dtype=np.int_)
        for i in range(_k):
            indexes[i, 0] = id_x
            indexes[i, 1] = _argsort[i]
        return distance_profiles[_argsort[:_k]], indexes
    else:
        # Apply exclusion zone to avoid neighboring matches
        top_k = np.zeros((_k, 2), dtype=np.int_) - exclusion_size
        top_k_dist = np.zeros((_k), dtype=np.float64)

        top_k[0, 0] = id_x
        top_k[0, 1] = _argsort[0]

        top_k_dist[0] = distance_profiles[_argsort[0]]

        n_inserted = 1
        i_current = 1

        while n_inserted < _k and i_current < _argsort.shape[0]:
            candidate_timestamp = _argsort[i_current]

            insert = True
            LB = candidate_timestamp >= (top_k[:, 1] - exclusion_size)
            UB = candidate_timestamp <= (top_k[:, 1] + exclusion_size)
            if np.any(UB & LB):
                insert = False

            if insert:
                top_k[n_inserted, 0] = id_x
                top_k[n_inserted, 1] = _argsort[i_current]
                top_k_dist[n_inserted] = distance_profiles[_argsort[i_current]]
                n_inserted += 1
            i_current += 1
        return top_k_dist[:n_inserted], top_k[:n_inserted]


def extract_top_k_and_threshold_from_distance_profiles(
    distance_profiles,
    k=1,
    threshold=np.inf,
    exclusion_size=None,
    inverse_distance=False,
):
    """
    Extract the best matches from a distance profile given k and threshold parameters.

    Parameters
    ----------
    distance_profiles : np.ndarray, 2D array of shape (n_cases, n_candidates)
        Precomputed distance profile. Can be a TypedList if n_candidates vary between
        cases.
    k : int
        Number of matches to returns
    threshold : float
        All matches below this threshold will be returned
    exclusion_size : int, optional
        The size of the exclusion zone used to prevent returning as top k candidates
        the ones that are close to each other (for example i and i+1).
        It is used to define a region between
        :math:`id_timestamp - exclusion_size` and
        :math:`id_timestamp + exclusion_size` which cannot be returned
        as best match if :math:`id_timestamp` was already selected. By default,
        the value None means that this is not used.
    inverse_distance : bool, optional
        Wheter to return the worst matches instead of the bests. The default is False.

    Returns
    -------
    Tuple(ndarray, ndarray)
        The first array, of shape ``(n_matches)``, contains the distance between
        the query and its best matches in X_. The second array, of shape
        ``(n_matches, 2)``, contains the indexes of these matches as
        ``(id_sample, id_timepoint)``. The corresponding match can be
        retrieved as ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

    """
    # This whole function could be optimized and maybe made in numba to avoid stepping
    # out of numba mode during distance computations

    n_cases_ = len(distance_profiles)

    id_timestamps = np.concatenate(
        [np.arange(distance_profiles[i].shape[0]) for i in range(n_cases_)]
    )
    id_samples = np.concatenate(
        [[i] * distance_profiles[i].shape[0] for i in range(n_cases_)]
    )

    distance_profiles = np.concatenate(distance_profiles)

    if inverse_distance:
        # To avoid div by 0 case
        distance_profiles += 1e-8
        distance_profiles[distance_profiles != np.inf] = (
            1 / distance_profiles[distance_profiles != np.inf]
        )

    if threshold != np.inf:
        distance_profiles[distance_profiles > threshold] = np.inf

    _argsort_1d = np.argsort(distance_profiles)
    _argsort = np.asarray(
        [
            [id_samples[_argsort_1d[i]], id_timestamps[_argsort_1d[i]]]
            for i in range(len(_argsort_1d))
        ],
        dtype=int,
    )

    if distance_profiles[distance_profiles <= threshold].shape[0] < k:
        _k = distance_profiles[distance_profiles <= threshold].shape[0]
        warnings.warn(
            f"Only {_k} matches are bellow the threshold of {threshold}, while"
            f" k={k}. The number of returned match will be {_k}.",
            stacklevel=2,
        )
    elif _argsort.shape[0] < k:
        _k = _argsort.shape[0]
        warnings.warn(
            f"The number of possible match is {_argsort.shape[0]}, but got"
            f" k={k}. The number of returned match will be {_k}.",
            stacklevel=2,
        )
    else:
        _k = k

    if exclusion_size is None:
        return distance_profiles[_argsort_1d[:_k]], _argsort[:_k]
    else:
        # Apply exclusion zone to avoid neighboring matches
        top_k = np.zeros((_k, 2), dtype=int)
        top_k_dist = np.zeros((_k), dtype=float)

        top_k[0] = _argsort[0, :]
        top_k_dist[0] = distance_profiles[_argsort_1d[0]]

        n_inserted = 1
        i_current = 1

        while n_inserted < _k and i_current < _argsort.shape[0]:
            candidate_sample, candidate_timestamp = _argsort[i_current]

            insert = True
            is_from_same_sample = top_k[:, 0] == candidate_sample
            if np.any(is_from_same_sample):
                LB = candidate_timestamp >= (
                    top_k[is_from_same_sample, 1] - exclusion_size
                )
                UB = candidate_timestamp <= (
                    top_k[is_from_same_sample, 1] + exclusion_size
                )
                if np.any(UB & LB):
                    insert = False

            if insert:
                top_k[n_inserted] = _argsort[i_current]
                top_k_dist[n_inserted] = distance_profiles[_argsort_1d[i_current]]
                n_inserted += 1
            i_current += 1
        return top_k_dist[:n_inserted], top_k[:n_inserted]
