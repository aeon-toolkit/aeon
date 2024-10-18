r"""Amercing dynamic time warping (ADTW) between two time series."""

__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.pointwise._squared import _univariate_squared_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def adtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> float:
    r"""Compute the ADTW distance between two time series.

    Amercing Dynamic Time Warping (ADTW) [1]_ is a variant of DTW that uses a
    explicit warping penalty to encourage or discourage warping. The warping
    penalty is a constant value that is added to the cost of warping. A high
    value will encourage the algorithm to warp less and if the value is low warping
    is more likely.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0.0 and 1.0
    warp_penalty: float, default=1.0
        Penalty for warping. A high value will mean less warping.

    Returns
    -------
    float
        ADTW distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Matthieu Herrmann, Geoffrey I. Webb: Amercing: An intuitive and effective
    constraint for dynamic time warping, Pattern Recognition, Volume 137, 2023.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import adtw_distance
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> adtw_distance(x, y) # 1D series
    783.0
    >>> x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 1, 0, 2, 0]])
    >>> y = np.array([[11, 12, 13, 14],[7, 8, 9, 20],[1, 3, 4, 5]] )
    >>> adtw_distance(x, y) # 2D series with 3 channels, unequal length
    565.0
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _adtw_distance(_x, _y, bounding_matrix, warp_penalty)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _adtw_distance(x, y, bounding_matrix, warp_penalty)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def adtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> np.ndarray:
    r"""Compute the ADTW cost matrix between two time series.

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
        is used. window is a percentage deviation, so if ``window = 0.1``,
        10% of the series length is the max warping allowed.
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    warp_penalty: float, default=1.0
        Penalty for warping. A high value will mean less warping.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        adtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import adtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> adtw_cost_matrix(x, y)
    array([[  0.,   2.,   7.,  17.,  34.,  60.,  97., 147., 212., 294.],
           [  2.,   0.,   2.,   7.,  17.,  34.,  60.,  97., 147., 212.],
           [  7.,   2.,   0.,   2.,   7.,  17.,  34.,  60.,  97., 147.],
           [ 17.,   7.,   2.,   0.,   2.,   7.,  17.,  34.,  60.,  97.],
           [ 34.,  17.,   7.,   2.,   0.,   2.,   7.,  17.,  34.,  60.],
           [ 60.,  34.,  17.,   7.,   2.,   0.,   2.,   7.,  17.,  34.],
           [ 97.,  60.,  34.,  17.,   7.,   2.,   0.,   2.,   7.,  17.],
           [147.,  97.,  60.,  34.,  17.,   7.,   2.,   0.,   2.,   7.],
           [212., 147.,  97.,  60.,  34.,  17.,   7.,   2.,   0.,   2.],
           [294., 212., 147.,  97.,  60.,  34.,  17.,   7.,   2.,   0.]])

    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _adtw_cost_matrix(_x, _y, bounding_matrix, warp_penalty)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _adtw_cost_matrix(x, y, bounding_matrix, warp_penalty)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _adtw_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, warp_penalty: float
) -> float:
    return _adtw_cost_matrix(x, y, bounding_matrix, warp_penalty)[
        x.shape[1] - 1, y.shape[1] - 1
    ]


@njit(cache=True, fastmath=True)
def _adtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, warp_penalty: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[:, i], y[:, j]
                ) + min(
                    cost_matrix[i, j + 1] + warp_penalty,
                    cost_matrix[i + 1, j] + warp_penalty,
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]


def adtw_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> np.ndarray:
    r"""Compute the ADTW pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the adtw pairwise distance between the instances of X is
        calculated.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    warp_penalty: float, default=1.0
        Penalty for warping. A high value will mean less warping.
        warp less and if value is low then will encourage algorithm to warp
        more.

    Returns
    -------
    np.ndarray
        ADTW pairwise matrix between the instances of X of shape
        ``(n_cases, n_cases)`` or between X and y of shape ``(n_cases,
        n_cases)``.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array and if y is not 1D, 2D or 3D arrays when passing y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import adtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> adtw_pairwise_distance(X)
    array([[  0.,  27., 108.],
           [ 27.,   0.,  27.],
           [108.,  27.,   0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> adtw_pairwise_distance(X, y)
    array([[300., 507., 768.],
           [147., 300., 507.],
           [ 48., 147., 300.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> adtw_pairwise_distance(X, y_univariate)
    array([[300.],
           [147.],
           [ 48.]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> adtw_pairwise_distance(X)
    array([[  0.,  44., 294.],
           [ 44.,   0.,  87.],
           [294.,  87.,   0.]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )
    if y is None:
        # To self
        return _adtw_pairwise_distance(
            _X, window, itakura_max_slope, warp_penalty, unequal_length
        )

    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _adtw_from_multiple_to_multiple_distance(
        _X, _y, window, itakura_max_slope, warp_penalty, unequal_length
    )


@njit(cache=True, fastmath=True)
def _adtw_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    itakura_max_slope: Optional[float],
    warp_penalty: float,
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
            distances[i, j] = _adtw_distance(x1, x2, bounding_matrix, warp_penalty)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _adtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    itakura_max_slope: Optional[float],
    warp_penalty: float,
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
            distances[i, j] = _adtw_distance(x1, y1, bounding_matrix, warp_penalty)
    return distances


@njit(cache=True, fastmath=True)
def adtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    warp_penalty: float = 1.0,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the ADTW alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    warp_penalty: float, default=1.0
        Penalty for warping. A high value will mean less warping.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The ADTW distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import adtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> adtw_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    cost_matrix = adtw_cost_matrix(x, y, window, itakura_max_slope, warp_penalty)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
