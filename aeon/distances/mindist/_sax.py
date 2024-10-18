__maintainer__ = []

from typing import Union

import numpy as np
from numba import njit, prange

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def mindist_sax_distance(
    x: np.ndarray, y: np.ndarray, breakpoints: np.ndarray, n: int
) -> float:
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

    References
    ----------
    .. [1] Camerra, Alessandro, et al. "isax 2.0: Indexing and mining one billion
    time series." 2010 IEEE international conference on data mining. IEEE, 2010.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import mindist_paa_sax_distance
    >>> from aeon.transformations.collection.dictionary_based import SAX
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> transform = SAX(n_segments=8, alphabet_size=8)
    >>> x_sax = transform.fit_transform(x).squeeze()
    >>> y_sax = transform.transform(y).squeeze()
    >>> dist = mindist_paa_sax_distance(
    ... x_sax, y_sax, transform.breakpoints, x.shape[-1]
    ... )
    """
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_sax_distance(x, y, breakpoints, n)
    raise ValueError("x and y must be 1D")


@njit(cache=True, fastmath=True)
def _univariate_sax_distance(
    x: np.ndarray, y: np.ndarray, breakpoints: np.ndarray, n: int
) -> float:
    dist = 0.0

    # The number of segments
    m = x.shape[0]

    # Compute the actual length of each segment in analogy to the PAA transform
    n_split = np.array_split(np.arange(n), m)

    for i in range(x.shape[0]):
        if np.abs(x[i] - y[i]) <= 1:
            continue
        else:
            dist += (
                n_split[i].shape[0]
                * (breakpoints[max(x[i], y[i]) - 1] - breakpoints[min(x[i], y[i])]) ** 2
            )

    return np.sqrt(dist)


def mindist_sax_pairwise_distance(
    X: np.ndarray, y: np.ndarray, breakpoints: np.ndarray, n: int
) -> np.ndarray:
    """Compute the SAX pairwise distance between a set of SAX representations.

    Parameters
    ----------
    X : np.ndarray
        A collection of SAX instances  of shape ``(n_instances, n_timepoints)``.
    y : np.ndarray
        A collection of SAX instances  of shape ``(n_instances, n_timepoints)``.
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
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )
    if y is None:
        return _sax_from_multiple_to_multiple_distance(_X, None, breakpoints, n)

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _sax_from_multiple_to_multiple_distance(_X, _y, breakpoints, n)


@njit(cache=True, fastmath=True, parallel=True)
def _sax_from_multiple_to_multiple_distance(
    X: np.ndarray, y: Union[np.ndarray, None], breakpoints: np.ndarray, n: int
) -> np.ndarray:
    if y is None:
        n_instances = X.shape[0]
        distances = np.zeros((n_instances, n_instances))

        for i in prange(n_instances):
            for j in prange(i + 1, n_instances):
                distances[i, j] = _univariate_sax_distance(X[i], X[j], breakpoints, n)
                distances[j, i] = distances[i, j]
    else:
        n_instances = X.shape[0]
        m_instances = y.shape[0]
        distances = np.zeros((n_instances, m_instances))

        for i in prange(n_instances):
            for j in prange(m_instances):
                distances[i, j] = _univariate_sax_distance(X[i], y[j], breakpoints, n)

    return distances
