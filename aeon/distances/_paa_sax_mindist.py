__maintainer__ = []

import numpy as np
from numba import njit, prange

from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def paa_sax_mindist(
    x_paa: np.ndarray, y_sax: np.ndarray, breakpoints: np.ndarray, n: int
) -> float:
    r"""Compute the PAA-SAX lower bounding distance between PAA and SAX representation.

    Parameters
    ----------
    x_paa : np.ndarray
        First PAA transform of the time series, univariate, shape ``(n_timepoints,)``
    y_sax : np.ndarray
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

    References
    ----------
    .. [1] Camerra, Alessandro, et al. "isax 2.0: Indexing and mining one billion
    time series." 2010 IEEE international conference on data mining. IEEE, 2010.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import paa_sax_mindist
    >>> from aeon.transformations.collection.dictionary_based import SAX
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> transform = SAX(n_segments=8, alphabet_size=8)
    >>> x_sax = transform.fit_transform(x).squeeze()
    >>> x_paa = transform._get_paa(x).squeeze()
    >>> y_sax = transform.transform(y).squeeze()
    >>> dist = paa_sax_mindist(x_paa, y_sax, transform.breakpoints, x.shape[-1])
    """
    if x_paa.ndim == 1 and y_sax.ndim == 1:
        return _univariate_PAA_SAX_distance(x_paa, y_sax, breakpoints, n)
    raise ValueError("x and y must be 1D")


@njit(cache=True, fastmath=True)
def _univariate_PAA_SAX_distance(
    x_paa: np.ndarray, y_sax: np.ndarray, breakpoints: np.ndarray, n: int
) -> float:
    dist = 0.0
    for i in range(x_paa.shape[0]):
        if y_sax[i] >= breakpoints.shape[0]:
            br_upper = np.inf
        else:
            br_upper = breakpoints[y_sax[i]]

        if y_sax[i] - 1 < 0:
            br_lower = -np.inf
        else:
            br_lower = breakpoints[y_sax[i] - 1]

        if br_lower > x_paa[i]:
            dist += (br_lower - x_paa[i]) ** 2
        elif br_upper < x_paa[i]:
            dist += (x_paa[i] - br_upper) ** 2

    m = x_paa.shape[0]
    return np.sqrt(n / m) * np.sqrt(dist)


@njit(cache=True, fastmath=True)
def sax_pairwise_distance(
    X: np.ndarray, y: np.ndarray, breakpoints: np.ndarray, n: int
) -> np.ndarray:
    """Compute the SAX pairwise distance between a set of SAX representations.

    Parameters
    ----------
    X : np.ndarray
        A collection of PAA instances  of shape ``(n_instances, n_timepoints)``.
    y : np.ndarray
        A collection of paa instances  of shape ``(n_instances, n_timepoints)``.
    breakpoints: np.ndarray
        The breakpoints of the SAX transformation
    n : int
        The original size of the time series

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        SAX pairwise matrix between the instances of X.

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
            return _paa_sax_from_multiple_to_multiple_distance(_X, breakpoints, n)
        raise ValueError("X must be a 2D array")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _paa_sax_from_multiple_to_multiple_distance(_x, _y, breakpoints, n)


@njit(cache=True, fastmath=True, parallel=True)
def _paa_sax_from_multiple_to_multiple_distance(
    X: np.ndarray, y: np.ndarray, breakpoints: np.ndarray, n: int
) -> np.ndarray:
    if y is None:
        n_instances = X.shape[0]
        distances = np.zeros((n_instances, n_instances))

        for i in prange(n_instances):
            for j in prange(i + 1, n_instances):
                distances[i, j] = _univariate_PAA_SAX_distance(
                    X[i], X[j], breakpoints, n
                )
                distances[j, i] = distances[i, j]
    else:
        n_instances = X.shape[0]
        m_instances = y.shape[0]
        distances = np.zeros((n_instances, m_instances))

        for i in prange(n_instances):
            for j in prange(m_instances):
                distances[i, j] = _univariate_PAA_SAX_distance(
                    X[i], y[j], breakpoints, n
                )

    return distances
