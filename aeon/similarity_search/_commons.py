"""Common utility functions for similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "_inverse_distance_profile",
]

import numpy as np
from numba import njit

from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD


@njit(cache=True, fastmath=True)
def _inverse_distance_profile(dist_profile: np.ndarray) -> np.ndarray:
    """
    Invert a distance profile for farthest neighbor search.

    Converts a distance profile into an inverted form where small distances
    become large and vice versa. This is useful for finding the farthest
    neighbors instead of the nearest neighbors.

    Parameters
    ----------
    dist_profile : np.ndarray, 1D array
        Distance profile to invert.

    Returns
    -------
    np.ndarray
        Inverted distance profile where ``result[i] = 1 / (dist_profile[i] + eps)``.
        The small epsilon (``AEON_NUMBA_STD_THRESHOLD``) prevents division by zero.
    """
    return 1.0 / (dist_profile + AEON_NUMBA_STD_THRESHOLD)
