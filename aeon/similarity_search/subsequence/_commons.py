"""Helper functions for subsequence similarity search."""

__maintainer__ = ["baraline"]

import numpy as np
from numba import njit
from scipy.signal import convolve


@njit(cache=True)
def _extract_top_k_from_dist_profile(
    dist_profile,
    k,
    threshold,
    allow_trivial_matches,
    exclusion_size,
):
    """
    Given a 2D distance profile, extract the top k lowest distances.

    Parameters
    ----------
    dist_profile : np.ndarray, shape = (n_cases, n_timepoints - length + 1)
        A 2D distance profile where each row corresponds to a case/series and
        columns are candidate positions. ``length`` is the size of the query
        used to compute the distance profiles.
    k : int
        Number of best matches to return.
    threshold : float
        A threshold on the distances of the best matches. To be returned, a candidate
        must have a distance below this threshold. This can reduce the number of
        returned matches to be below ``k``.
    allow_trivial_matches : bool
        Whether to allow returning matches that are in the same neighborhood by
        ignoring the exclusion zone defined by the ``exclusion_size`` parameter.
        If False, the exclusion zone is applied within each series.
    exclusion_size : int
        The size of the exclusion zone to apply when ``allow_trivial_matches`` is
        False. It is applied on both sides of existing matches (+/- their indexes)
        within the same series.

    Returns
    -------
    top_k_indexes : np.ndarray, shape = (n_matches, 2)
        The indexes of the best matches as ``(i_case, i_timestep)`` pairs.
    top_k_distances : np.ndarray, shape = (n_matches,)
        The distances of the best matches.

    """
    n_cases, n_candidates = dist_profile.shape
    n_total = n_cases * n_candidates

    top_k_indexes = np.zeros((k, 2), dtype=np.int64)
    top_k_distances = np.full(k, np.inf, dtype=np.float64)

    # Track exclusion zones per case: lb and ub arrays for each found match
    # We store (case_idx, lb, ub) for each match to check exclusions
    exclusion_case = np.zeros(k, dtype=np.int64)
    exclusion_lb = np.full(k, -1.0)
    exclusion_ub = np.full(k, np.inf)

    # Flatten for efficient searching
    flat_profile = dist_profile.ravel()
    mask = np.ones(n_total, dtype=np.bool_)
    remaining_indices = np.arange(n_total)

    _current_k = 0

    if not allow_trivial_matches:
        while _current_k < k and np.any(mask):
            available_indices = remaining_indices[mask]
            search_k = min(k, len(available_indices))
            if search_k == 0:
                break

            # Find candidates with smallest distances
            partitioned = available_indices[
                np.argpartition(flat_profile[available_indices], search_k - 1)[
                    :search_k
                ]
            ]
            sorted_indexes = partitioned[np.argsort(flat_profile[partitioned])]

            for flat_idx in sorted_indexes:
                i_case = flat_idx // n_candidates
                i_ts = flat_idx % n_candidates

                # Check if in any exclusion zone (same case only)
                in_exclusion = False
                for j in range(_current_k):
                    if exclusion_case[j] == i_case:
                        if i_ts >= exclusion_lb[j] and i_ts <= exclusion_ub[j]:
                            in_exclusion = True
                            break

                if in_exclusion:
                    # Skip this candidate and test the next one
                    continue

                if flat_profile[flat_idx] > threshold:
                    # Distances are sorted, so we can break early
                    break

                top_k_indexes[_current_k, 0] = i_case
                top_k_indexes[_current_k, 1] = i_ts
                top_k_distances[_current_k] = flat_profile[flat_idx]

                # Store exclusion zone for this case
                exclusion_case[_current_k] = i_case
                exclusion_lb[_current_k] = max(i_ts - exclusion_size, 0)
                exclusion_ub[_current_k] = min(i_ts + exclusion_size, n_candidates - 1)
                _current_k += 1

                if _current_k == k:
                    break

            # Mark processed indices
            for idx in sorted_indexes:
                mask[idx] = False
    else:
        # Trivial matches allowed - just find k smallest globally
        search_k = min(k, n_total)
        partitioned = np.argpartition(flat_profile, search_k - 1)[:search_k]
        sorted_indexes = partitioned[np.argsort(flat_profile[partitioned])]

        for flat_idx in sorted_indexes:
            if flat_profile[flat_idx] <= threshold:
                i_case = flat_idx // n_candidates
                i_ts = flat_idx % n_candidates
                top_k_indexes[_current_k, 0] = i_case
                top_k_indexes[_current_k, 1] = i_ts
                top_k_distances[_current_k] = flat_profile[flat_idx]
                _current_k += 1
            else:
                break

    return top_k_indexes[:_current_k], top_k_distances[:_current_k]


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
