__maintainer__ = []

import numpy as np
from numba import njit

# from aeon.distances._utils import reshape_pairwise_to_multiple


# @njit(cache=True, fastmath=True)
def dft_sfa_mindist(
    x_dft: np.ndarray, y_sfa: np.ndarray, breakpoints: np.ndarray
) -> float:
    r"""Compute the DFT-SFA lower bounding distance between DFT and SFA representation.

    Parameters
    ----------
    x_dft : np.ndarray
        First DFT transform of the time series, univariate, shape ``(n_timepoints,)``
    y_sfa : np.ndarray
        Second SFA transform of the time series, univariate, shape ``(n_timepoints,)``
    breakpoints: np.ndarray
        The breakpoints of the SFA transformation
    Returns
    -------
    float
        SFA lower bounding distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    """
    x_dft = x_dft.squeeze()
    y_sfa = y_sfa.squeeze()

    if x_dft.ndim == 1 and y_sfa.ndim == 1:
        return _univariate_DFT_SFA_distance(x_dft, y_sfa, breakpoints)
    raise ValueError("x and y must be 1D")


@njit(cache=True, fastmath=True)
def _univariate_DFT_SFA_distance(
    x_dft: np.ndarray, y_sfa: np.ndarray, breakpoints: np.ndarray
) -> float:
    dist = 0.0
    for i in range(x_dft.shape[0]):
        if y_sfa[i] >= breakpoints.shape[-1]:
            br_upper = np.inf
        else:
            br_upper = breakpoints[i, y_sfa[i]]

        if y_sfa[i] - 1 < 0:
            br_lower = -np.inf
        else:
            br_lower = breakpoints[i, y_sfa[i] - 1]

        if br_lower > x_dft[i]:
            dist += (br_lower - x_dft[i]) ** 2
        elif br_upper < x_dft[i]:
            dist += (x_dft[i] - br_upper) ** 2

    return np.sqrt(2 * dist)


# @njit(cache=True, fastmath=True)
# def sfa_pairwise_distance(X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
#     """Compute the SFA pairwise distance between a set of SFA representations.
#
#     Parameters
#     ----------
#     X : np.ndarray
#         A collection of SFA instances  of shape ``(n_instances, n_timepoints)``.
#
#     Returns
#     -------
#     np.ndarray (n_instances, n_instances)
#         SFA pairwise matrix between the instances of X.
#
#     Raises
#     ------
#     ValueError
#         If X is not 2D array when only passing X.
#         If X and y are not 1D, 2D arrays when passing both X and y.
#
#     Examples
#     --------
#     """
#     if y is None:
#         # To self
#         if X.ndim == 2:
#             _X = X.reshape((X.shape[0], 1, X.shape[1]))
#             return _sfa_from_multiple_to_multiple_distance(_X)
#         raise ValueError("X must be a 2D array")
#     # TODO needed???
#     _x, _y = reshape_pairwise_to_multiple(X, y)
#     return _sfa_from_multiple_to_multiple_distance(_x, _y)
#
#
# @njit(cache=True, fastmath=True)
# def _sfa_from_multiple_to_multiple_distance(
#         X: np.ndarray,
#         breakpoints: np.ndarray,
#         n: int
#     ) -> np.ndarray:
#     n_instances = X.shape[0]
#     distances = np.zeros((n_instances, n_instances))
#
#     for i in range(n_instances):
#         for j in range(i + 1, n_instances):
#             distances[i, j] = _univariate_SFA_distance(X[i], X[j], breakpoints, n)
#             distances[j, i] = distances[i, j]
#
#     return distances
#
#
# @njit(cache=True, fastmath=True)
# def _sfa_from_multiple_to_multiple_distance(
#         x: np.ndarray, y: np.ndarray,
#         breakpoints: np.ndarray,
#         n: int
# ) -> np.ndarray:
#     n_instances = x.shape[0]
#     m_instances = y.shape[0]
#     distances = np.zeros((n_instances, m_instances))
#
#     for i in range(n_instances):
#         for j in range(m_instances):
#             distances[i, j] = _univariate_SFA_distance(x[i], y[j], breakpoints, n)
#     return distances
