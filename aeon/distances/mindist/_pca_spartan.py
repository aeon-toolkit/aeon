__maintainer__ = []

from typing import Union

import numpy as np
from numba import njit, prange

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def mindist_pca_spartan_distance(
    x_pca: np.ndarray,
    y_spartan: np.ndarray,
    breakpoints: np.ndarray,
    squared_lower_bound: float = np.inf,
) -> float:
    r"""Compute PCA-SPARTAN lower bounding dist between PCA and SPARTAN representation.

    Parameters
    ----------
    x_pca : np.ndarray
        PCA transform of the time series, univariate, shape ``(n_timepoints,)``
    y_spartan : np.ndarray
        SPARTAN transform of the time series, univariate, shape ``(n_timepoints,)``
    breakpoints: np.ndarray
        The breakpoints of the SPARTAN transformation
    squared_lower_bound : float
        Used for early stopping distance computations. Once the distance exceeds the
        squared lower bound, infinity is returned. Commonly used when searching
        for epsilon-range or k-nearest neighbors.

    Returns
    -------
    float
        SPARTAN lower bounding distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] ...

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import mindist_pca_spartan_distance
    >>> from aeon.transformations.collection.dictionary_based import SPARTAN
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> transform = SPARTAN()
    >>> x_spartan, _ = transform.fit_transform(x)
    >>> _, y_pca = transform.transform(y)
    >>> dist = mindist_pca_spartan_distance(y_pca, x_spartan, transform.breakpoints)
    """
    if x_pca.ndim == 1 and y_spartan.ndim == 1:
        return _univariate_pca_spartan_distance(
            x_pca, y_spartan, breakpoints, squared_lower_bound
        )
    raise ValueError("x and y must be 1D")


@njit(cache=True, fastmath=True)
def _univariate_pca_spartan_distance(
    x_pca: np.ndarray,
    y_spartan: np.ndarray,
    breakpoints: np.ndarray,
    squared_lower_bound: float = np.inf,
) -> float:
    dist = 0.0
    for i in range(x_pca.shape[0]):
        if y_spartan[i] >= breakpoints.shape[-1]:
            br_upper = np.inf
        else:
            br_upper = breakpoints[i, y_spartan[i]]

        if y_spartan[i] - 1 < 0:
            br_lower = -np.inf
        else:
            br_lower = breakpoints[i, y_spartan[i] - 1]

        if br_lower > x_pca[i]:
            dist += (br_lower - x_pca[i]) ** 2
        elif br_upper < x_pca[i]:
            dist += (x_pca[i] - br_upper) ** 2

        if dist > squared_lower_bound:
            return np.inf

    return np.sqrt(dist)


def mindist_pca_spartan_pairwise_distance(
    X: np.ndarray, y: np.ndarray, breakpoints: np.ndarray
) -> np.ndarray:
    """Compute the PCA SPARTAN pairwise dist between a set of SPARTAN representations.

    Parameters
    ----------
    X : np.ndarray
        A collection of PCA instances  of shape ``(n_instances, n_timepoints)``.
    y : np.ndarray
        A collection of SPARTAN instances  of shape ``(n_instances, n_timepoints)``.
    breakpoints: np.ndarray
        The breakpoints of the SAX transformation

    Returns
    -------
    np.ndarray (n_instances, n_instances)
        SPARTAN pairwise matrix between the instances of X.

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
        return _pca_spartan_from_multiple_to_multiple_distance(_X, None, breakpoints)

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _pca_spartan_from_multiple_to_multiple_distance(_X, _y, breakpoints)


@njit(cache=True, fastmath=True, parallel=True)
def _pca_spartan_from_multiple_to_multiple_distance(
    X: np.ndarray, y: Union[np.ndarray, None], breakpoints: np.ndarray
) -> np.ndarray:
    if y is None:
        n_instances = X.shape[0]
        distances = np.zeros((n_instances, n_instances))

        for i in prange(n_instances):
            for j in prange(i + 1, n_instances):
                distances[i, j] = _univariate_pca_spartan_distance(
                    X[i], X[j], breakpoints
                )
                distances[j, i] = distances[i, j]
    else:
        n_instances = X.shape[0]
        m_instances = y.shape[0]
        distances = np.zeros((n_instances, m_instances))

        for i in prange(n_instances):
            for j in prange(m_instances):
                distances[i, j] = _univariate_pca_spartan_distance(
                    X[i], y[j], breakpoints
                )

    return distances
