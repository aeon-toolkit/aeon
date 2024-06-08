__maintainer__ = []

import numpy as np
from numba import njit

# from aeon.distances._utils import reshape_pairwise_to_multiple


# @njit(cache=True, fastmath=True)
def sax_mindist(x: np.ndarray, y: np.ndarray, breakpoints: np.ndarray, n: int) -> float:
    r"""Compute the SAX lower bounding distance between two SAX representations.

    Parameters
    ----------
    x : np.ndarray
        First SAX transform of the time series, univariate, shape ``(n_timepoints,)``
    y : np.ndarray
        Second SAX transform of the time series, univariate, shape ``(n_timepoints,)``
    breakpoints: np.ndarray
        The breakpoints of the SAX transformation
    n : int
        The original size of the time series

    Returns
    -------
    float
        SAX lower bounding distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    """
    x = np.squeeze(x)
    y = np.squeeze(y)

    if x.ndim == 1 and y.ndim == 1:
        return _univariate_SAX_distance(x, y, breakpoints, n)
    raise ValueError("x and y must be 1D")


@njit(cache=True, fastmath=True)
def _univariate_SAX_distance(
    x: np.ndarray, y: np.ndarray, breakpoints: np.ndarray, n: int
) -> float:
    dist = 0.0
    for i in range(x.shape[0]):
        if np.abs(x[i] - y[i]) <= 1:
            continue
        else:
            dist += (
                breakpoints[max(x[i], y[i]) - 1] - breakpoints[min(x[i], y[i])]
            ) ** 2

    m = x.shape[0]
    return np.sqrt(n / m) * np.sqrt(dist)


# @njit(cache=True, fastmath=True)
# def sax_pairwise_distance(X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
#     """Compute the SAX pairwise distance between a set of SAX representations.
#
#     Parameters
#     ----------
#     X : np.ndarray
#         A collection of SAX instances  of shape ``(n_instances, n_timepoints)``.
#
#     Returns
#     -------
#     np.ndarray (n_instances, n_instances)
#         SAX pairwise matrix between the instances of X.
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
#             return _sax_from_multiple_to_multiple_distance(_X)
#         raise ValueError("X must be a 2D array")
#     # TODO needed???
#     _x, _y = reshape_pairwise_to_multiple(X, y)
#     return _sax_from_multiple_to_multiple_distance(_x, _y)
#
#
# @njit(cache=True, fastmath=True)
# def _sax_from_multiple_to_multiple_distance(
#         X: np.ndarray,
#         breakpoints: np.ndarray,
#         n: int
#     ) -> np.ndarray:
#     n_instances = X.shape[0]
#     distances = np.zeros((n_instances, n_instances))
#
#     for i in range(n_instances):
#         for j in range(i + 1, n_instances):
#             distances[i, j] = _univariate_SAX_distance(X[i], X[j], breakpoints, n)
#             distances[j, i] = distances[i, j]
#
#     return distances
#
#
# @njit(cache=True, fastmath=True)
# def _sax_from_multiple_to_multiple_distance(
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
#             distances[i, j] = _univariate_SAX_distance(x[i], y[j], breakpoints, n)
#     return distances
