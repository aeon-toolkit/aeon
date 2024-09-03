"""Numba-accelerated discrete wavelet transformations (DFTs)."""

import numpy as np
from numba import njit
from numba.typed import List

__maintainer__ = []
__all__ = [
    "haar_transform",
    "multilevel_haar_transform",
]


def multilevel_haar_transform(
    x: np.ndarray, levels: int = 1
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Perform the multilevel discrete Haar wavelet transform on a given signal.

    Captures the approximate and detail coefficients per level. The approximate
    coefficients contain one more element than the detail coefficients.

    Parameters
    ----------
    x : np.ndarray
        The input signal.
    levels : int
        The number of levels to perform the Haar wavelet transform.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        The approximate and detail coefficients per level.
    """
    N = len(x)
    max_levels = np.floor(np.log2(N))
    if levels > max_levels:
        raise ValueError(
            f"The level ({levels}) must be less than log_2(N) ({max_levels})."
        )

    res_approx, res_detail = _haar_transform_iterative(x, levels)
    return res_approx, res_detail


@njit(cache=True, fastmath=True)
def haar_transform(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform the discrete Haar wavelet transform on a given signal.

    Parameters
    ----------
    x : np.ndarray
        The input signal.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The approximate and detail coefficients.
    """
    approx, detail = _haar_transform_iterative(x, levels=1)
    return approx[-1], detail[-1]


@njit(cache=True, fastmath=True)
def _haar_transform_iterative(
    x: np.ndarray, levels: int
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    # initialize
    l_approx = List()
    l_approx.append(x)
    l_detail = List()

    for _ in range(1, levels + 1):
        approx = l_approx[-1]
        l_approx.append(_haar_approx_coefficients(approx))
        l_detail.append(_haar_detail_coefficients(approx))

    return l_approx, l_detail


@njit(cache=True, fastmath=True)
def _haar_approx_coefficients(arr: np.ndarray) -> np.ndarray:
    """Get the approximate coefficients at a given level."""
    if len(arr) == 1:
        return np.array([arr[0]])

    N = int(np.floor(len(arr) / 2))
    new = np.empty(N, dtype=arr.dtype)
    for i in range(N):
        new[i] = (arr[2 * i] + arr[2 * i + 1]) / np.sqrt(2)
    return new


@njit(cache=True, fastmath=True)
def _haar_detail_coefficients(arr: np.ndarray) -> np.ndarray:
    """Get the detail coefficients at a given level."""
    if len(arr) == 1:
        return np.array([arr[0]])

    N = int(np.floor(len(arr) / 2))
    new = np.empty(N, dtype=arr.dtype)
    for i in range(N):
        new[i] = (arr[2 * i] - arr[2 * i + 1]) / np.sqrt(2)
    return new
