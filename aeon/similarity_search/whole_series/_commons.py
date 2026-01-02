"""Common utility functions for whole series similarity search."""

__maintainer__ = ["baraline"]

import numpy as np
from numba import njit

# Re-export commonly used functions from the parent commons module
from aeon.similarity_search._commons import _inverse_distance_profile  # noqa: F401


@njit(cache=True)
def _extract_top_k_from_dist_profile(
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
