__maintainer__ = []

import numpy as np
from numba import njit, prange

from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def sfa_mindist(x: np.ndarray, y: np.ndarray, breakpoints: np.ndarray) -> float:
    r"""Compute the SFA lower bounding distance between two SFA representations.

    Parameters
    ----------
    x : np.ndarray
        First SFA transform of the time series, univariate, shape ``(n_timepoints,)``
    y : np.ndarray
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
    >>> from aeon.distances import sfa_mindist
    >>> from aeon.transformations.collection.dictionary_based import SFAFast
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> transform = SFAFast(
    ...    word_length=8,
    ...    alphabet_size=8,
    ...    window_size=x.shape[-1],
    ...    norm=True,
    ...    lower_bounding_distances=True   # This must be set!
    ... )
    >>> transform.fit(x)
    SFAFast(...)
    >>> x_sfa = transform.transform_words(x).squeeze()
    >>> y_sfa = transform.transform_words(y).squeeze()
    >>> dist = sfa_mindist(x_sfa, y_sfa, transform.breakpoints)
    """
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_SFA_distance(x, y, breakpoints)
    raise ValueError("x and y must be 1D")


@njit(cache=True, fastmath=True)
def _univariate_SFA_distance(
    x: np.ndarray, y: np.ndarray, breakpoints: np.ndarray
) -> float:
    dist = 0.0
    for i in range(x.shape[0]):
        if np.abs(x[i] - y[i]) <= 1:
            continue
        else:
            dist += (
                breakpoints[i, max(x[i], y[i]) - 1] - breakpoints[i, min(x[i], y[i])]
            ) ** 2

    return np.sqrt(2 * dist)


@njit(cache=True, fastmath=True)
def sfa_pairwise_distance(
    X: np.ndarray, y: np.ndarray, breakpoints: np.ndarray
) -> np.ndarray:
    """Compute the SFA pairwise distance between a set of SFA representations.

    Parameters
    ----------
    X : np.ndarray
        A collection of SFA instances  of shape ``(n_instances, n_timepoints)``.
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
    if y is None:
        # To self
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _sfa_from_multiple_to_multiple_distance(_X, None, breakpoints)
        raise ValueError("X must be a 2D array")

    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _sfa_from_multiple_to_multiple_distance(_x, _y, breakpoints)


@njit(cache=True, fastmath=True, parallel=True)
def _sfa_from_multiple_to_multiple_distance(
    X: np.ndarray, y: np.ndarray, breakpoints: np.ndarray
) -> np.ndarray:
    if y is None:
        n_instances = X.shape[0]
        distances = np.zeros((n_instances, n_instances))

        for i in prange(n_instances):
            for j in prange(i + 1, n_instances):
                distances[i, j] = _univariate_SFA_distance(X[i], X[j], breakpoints)
                distances[j, i] = distances[i, j]
    else:
        n_instances = X.shape[0]
        m_instances = y.shape[0]
        distances = np.zeros((n_instances, m_instances))

        for i in prange(n_instances):
            for j in prange(m_instances):
                distances[i, j] = _univariate_SFA_distance(X[i], y[j], breakpoints)

    return distances
