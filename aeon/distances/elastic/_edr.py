"""Edit distance for real sequences (EDR) between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.pointwise._euclidean import _univariate_euclidean_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def edr_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the EDR distance between two time series.

    Edit Distance on Real Sequences (EDR) was proposed as an adaptation of standard
    edit distance on discrete sequences in [1]_, specifically for distances between
    trajectories. Like LCSS, EDR uses a distance threshold to define when two
    elements of a series match. However, rather than simply count matches and look
    for the longest sequence, EDR applies a (constant) penalty for non-matching elements
    where gaps are inserted to create an optimal alignment.

    .. math::
        if \;\; |ai − bj | < ϵ\\
            c &= 0\\
        else\\
            c &= 1\\
        match  &=  D_{i-1,j-1}+ c)\\
        delete &=   D_{i-1,j}+ d({x_{i},g})\\
        insert &=  D_{i-1,j-1}+ d({g,y_{j}})\\
        D_{i,j} &= min(match,insert, delete)

    EDR computes the minimum number of elements (as a percentage) that must be removed
    from x and y so that the sum of the distance between the remaining signal elements
    lies within the tolerance (epsilon). EDR was originally proposed in [1]_.

    The value returned will be between 0 and 1 per time series. The value will
    represent as a percentage of elements that must be removed for the time series to
    be an exact match.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        EDR distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.


    References
    ----------
    .. [1] Chen L, Ozsu MT, Oria V: Robust and fast similarity search for moving
    object trajectories. In: Proceedings of the ACM SIGMOD International Conference
    on Management of Data, 2005

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> edr_distance(x, y)
    1.0
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _edr_distance(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _edr_distance(x, y, bounding_matrix, epsilon)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def edr_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the EDR cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    epsilon : float, default=None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        EDR cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> edr_cost_matrix(x, y)
    array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 0., 1., 2., 2., 2., 2., 2., 2., 2.],
           [1., 1., 0., 1., 2., 3., 3., 3., 3., 3.],
           [1., 2., 1., 0., 1., 2., 3., 4., 4., 4.],
           [1., 2., 2., 1., 0., 1., 2., 3., 4., 5.],
           [1., 2., 3., 2., 1., 0., 1., 2., 3., 4.],
           [1., 2., 3., 3., 2., 1., 0., 1., 2., 3.],
           [1., 2., 3., 4., 3., 2., 1., 0., 1., 2.],
           [1., 2., 3., 4., 4., 3., 2., 1., 0., 1.],
           [1., 2., 3., 4., 5., 4., 3., 2., 1., 0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _edr_cost_matrix(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _edr_cost_matrix(x, y, bounding_matrix, epsilon)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _edr_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    epsilon: Optional[float] = None,
) -> float:
    distance = _edr_cost_matrix(x, y, bounding_matrix, epsilon)[
        x.shape[1] - 1, y.shape[1] - 1
    ]
    return float(distance / max(x.shape[1], y.shape[1]))


@njit(cache=True, fastmath=True)
def _edr_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    epsilon: Optional[float] = None,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    if epsilon is None:
        epsilon = float(max(np.std(x), np.std(y))) / 4

    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)

    for i in range(1, x_size + 1):
        if bounding_matrix[i - 1, 0]:
            cost_matrix[i, 0] = 0
    for j in range(y_size):
        if bounding_matrix[0, j - 1]:
            cost_matrix[0, j] = 0
    cost_matrix[0, 0] = 0

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                if _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]) < epsilon:
                    cost = 0
                else:
                    cost = 1
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + cost,
                    cost_matrix[i - 1, j] + 1,
                    cost_matrix[i, j - 1] + 1,
                )
    return cost_matrix[1:, 1:]


def edr_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the pairwise EDR distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the edr pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        EDR pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> edr_pairwise_distance(X)
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> edr_pairwise_distance(X, y)
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> edr_pairwise_distance(X, y_univariate)
    array([[1.],
           [1.],
           [1.]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> edr_pairwise_distance(X)
    array([[0.  , 0.75, 0.6 ],
           [0.75, 0.  , 0.8 ],
           [0.6 , 0.8 , 0.  ]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )

    if y is None:
        # To self
        return _edr_pairwise_distance(
            _X, window, epsilon, itakura_max_slope, unequal_length
        )

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _edr_from_multiple_to_multiple_distance(
        _X, _y, window, epsilon, itakura_max_slope, unequal_length
    )


@njit(cache=True, fastmath=True)
def _edr_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    epsilon: Optional[float],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _edr_distance(x1, x2, bounding_matrix, epsilon)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _edr_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    epsilon: Optional[float],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _edr_distance(x1, y1, bounding_matrix, epsilon)
    return distances


@njit(cache=True, fastmath=True)
def edr_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    epsilon: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the EDR alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The EDR distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import edr_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> edr_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 0.25)
    """
    cost_matrix = edr_cost_matrix(x, y, window, epsilon, itakura_max_slope)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1] / max(x.shape[-1], y.shape[-1]),
    )
