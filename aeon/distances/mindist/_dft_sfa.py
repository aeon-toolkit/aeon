__maintainer__ = []

from typing import Union

import numpy as np
from numba import njit, prange

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def mindist_dft_sfa_distance(
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

    References
    ----------
    .. [1] Schäfer, Patrick, and Mikael Högqvist. "SFA: a symbolic fourier approximation
    and  index for similarity search in high dimensional datasets." Proceedings of the
    15th international conference on extending database technology. 2012.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import mindist_dft_sfa_distance
    >>> from aeon.transformations.collection.dictionary_based import SFAWhole
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> transform = SFAWhole(
    ...    word_length=8,
    ...    alphabet_size=8,
    ...    norm=True,
    ... )
    >>> x_sfa, _ = transform.fit_transform(x)
    >>> _, y_dft = transform.transform(y)
    >>> dist = mindist_dft_sfa_distance(y_dft, x_sfa, transform.breakpoints)
    """
    if x_dft.ndim == 1 and y_sfa.ndim == 1:
        return _univariate_dft_sfa_distance(x_dft, y_sfa, breakpoints)
    raise ValueError("x and y must be 1D")


@njit(cache=True, fastmath=True)
def _univariate_dft_sfa_distance(
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


def mindist_dft_sfa_pairwise_distance(
    X: np.ndarray, y: np.ndarray, breakpoints: np.ndarray
) -> np.ndarray:
    """Compute the DFT SFA pairwise distance between a set of SFA representations.

    Parameters
    ----------
    X : np.ndarray
        A collection of DFT instances  of shape ``(n_instances, n_timepoints)``.
    y : np.ndarray
        A collection of SFA instances  of shape ``(n_instances, n_timepoints)``.
    breakpoints: np.ndarray
        The breakpoints of the SAX transformation

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        SFA pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D array when only passing X.
        If X and y are not 1D, 2D arrays when passing both X and y.
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )
    if y is None:
        return _dft_sfa_from_multiple_to_multiple_distance(_X, None, breakpoints)

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _dft_sfa_from_multiple_to_multiple_distance(_X, _y, breakpoints)


@njit(cache=True, fastmath=True, parallel=True)
def _dft_sfa_from_multiple_to_multiple_distance(
    X: np.ndarray, y: Union[np.ndarray, None], breakpoints: np.ndarray
) -> np.ndarray:
    if y is None:
        n_instances = X.shape[0]
        distances = np.zeros((n_instances, n_instances))

        for i in prange(n_instances):
            for j in prange(i + 1, n_instances):
                distances[i, j] = _univariate_dft_sfa_distance(X[i], X[j], breakpoints)
                distances[j, i] = distances[i, j]
    else:
        n_instances = X.shape[0]
        m_instances = y.shape[0]
        distances = np.zeros((n_instances, m_instances))

        for i in prange(n_instances):
            for j in prange(m_instances):
                distances[i, j] = _univariate_dft_sfa_distance(X[i], y[j], breakpoints)

    return distances
