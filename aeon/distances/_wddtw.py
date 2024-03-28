"""Weighted derivative dynamic time warping (wddtw) distance between two series."""

__maintainer__ = []

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from aeon.distances._alignment_paths import compute_min_return_path
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._ddtw import average_of_slope
from aeon.distances._utils import reshape_pairwise_to_multiple
from aeon.distances._wdtw import _wdtw_cost_matrix, _wdtw_distance


@njit(cache=True, fastmath=True)
def wddtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.05,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the WDDTW distance between two time series.

    Weighted derivative dynamic time warping (WDDTW) Takes the first order derivative,
    then applies `_weighted_cost_matrix` to find WDTW distance. WDDTW was first
    proposed in [1]_ as an extension of DDTW. By adding a weight to the derivative it
    means the alignment isn't only considering the shape of the time series, but also
    the phase.

    Formally the derivative is calculated as:

    .. math::
        d_{i}(x) = \frac{{}(x_{i} - x_{i-1} + ((x_{i+1} - x_{i-1}/2)}{2}

    where :math:`x` is the original time series and :math:`d_x` is the derived time
    series.

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
    g : float, default=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        WDDTW distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If n_timepoints or m_timepoints are less than 2.

    References
    ----------
    .. [1] Young-Seon Jeong, Myong K. Jeong, Olufemi A. Omitaomu, Weighted dynamic time
    warping for time series classification, Pattern Recognition, Volume 44, Issue 9,
    2011, Pages 2231-2240, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2010.09.022.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[42, 23, 21, 55, 1, 19, 33, 34, 29, 19]])
    >>> round(wddtw_distance(x, y))
    981
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = average_of_slope(x.reshape((1, x.shape[0])))
        _y = average_of_slope(y.reshape((1, y.shape[0])))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _wdtw_distance(_x, _y, bounding_matrix, g)
    if x.ndim == 2 and y.ndim == 2:
        _x = average_of_slope(x)
        _y = average_of_slope(y)
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _wdtw_distance(_x, _y, bounding_matrix, g)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def wddtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.05,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the WDDTW cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints,)
        First time series.
    y : np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints,)
        Second time series.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g : float, default=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints_x, n_timepoints_y)
        WDDTW cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If n_timepoints or m_timepoints are less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_cost_matrix
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> wddtw_cost_matrix(x, y)
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = average_of_slope(x.reshape((1, x.shape[0])))
        _y = average_of_slope(y.reshape((1, y.shape[0])))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _wdtw_cost_matrix(_x, _y, bounding_matrix, g)
    if x.ndim == 2 and y.ndim == 2:
        _x = average_of_slope(x)
        _y = average_of_slope(y)
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _wdtw_cost_matrix(_x, _y, bounding_matrix, g)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def wddtw_pairwise_distance(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    window: Optional[float] = None,
    g: float = 0.05,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the WDDTW pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the wddtw pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g : float, default=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.
        If n_timepoints is less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[49, 58, 61]], [[73, 82, 99]]])
    >>> wddtw_pairwise_distance(X)
    array([[ 0.        , 20.86095125, 49.37503255],
           [20.86095125,  0.        ,  6.04844149],
           [49.37503255,  6.04844149,  0.        ]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[19, 12, 39]],[[40, 51, 69]], [[79, 28, 91]]])
    >>> y = np.array([[[110, 15, 123]],[[14, 150, 116]], [[9917, 118, 29]]])
    >>> wddtw_pairwise_distance(X, y)
    array([[1.03345029e+03, 4.17910276e+03, 2.68408251e+07],
           [1.60419481e+03, 3.21952986e+03, 2.69227971e+07],
           [2.33574763e+02, 6.64390438e+03, 2.66663693e+07]])

    >>> X = np.array([[[10, 22, 399]],[[41, 500, 1316]], [[117, 18, 9]]])
    >>> y_univariate = np.array([100, 11, 199])
    >>> wddtw_pairwise_distance(X, y_univariate)
    array([[  7469.9486745 ],
           [159295.70501427],
           [  1590.15378267]])
    """
    if y is None:
        # To self
        if X.ndim == 3:
            return _wddtw_pairwise_distance(X, window, g, itakura_max_slope)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _wddtw_pairwise_distance(_X, window, g, itakura_max_slope)
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _wddtw_from_multiple_to_multiple_distance(
        _x, _y, window, g, itakura_max_slope
    )


@njit(cache=True, fastmath=True)
def _wddtw_pairwise_distance(
    X: np.ndarray, window: Optional[float], g: float, itakura_max_slope: Optional[float]
) -> np.ndarray:
    n_cases = X.shape[0]
    distances = np.zeros((n_cases, n_cases))
    bounding_matrix = create_bounding_matrix(
        X.shape[2] - 2, X.shape[2] - 2, window, itakura_max_slope
    )

    X_average_of_slope = np.zeros((n_cases, X.shape[1], X.shape[2] - 2))
    for i in range(n_cases):
        X_average_of_slope[i] = average_of_slope(X[i])

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = _wdtw_distance(
                X_average_of_slope[i], X_average_of_slope[j], bounding_matrix, g
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _wddtw_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float],
    g: float,
    itakura_max_slope: Optional[float],
) -> np.ndarray:
    n_cases = x.shape[0]
    m_cases = y.shape[0]
    distances = np.zeros((n_cases, m_cases))
    bounding_matrix = create_bounding_matrix(
        x.shape[2], y.shape[2], window, itakura_max_slope
    )

    # Derive the arrays before so that we don't have to redo every iteration
    derive_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] - 2))
    for i in range(x.shape[0]):
        derive_x[i] = average_of_slope(x[i])

    derive_y = np.zeros((y.shape[0], y.shape[1], y.shape[2] - 2))
    for i in range(y.shape[0]):
        derive_y[i] = average_of_slope(y[i])

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = _wdtw_distance(
                derive_x[i], derive_y[j], bounding_matrix, g
            )
    return distances


@njit(cache=True, fastmath=True)
def wddtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    g: float = 0.05,
    itakura_max_slope: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the WDDTW alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g : float, default=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference. Default is 0.05.
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
        The wddtw distance betweeen the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If n_timepoints or m_timepoints are less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import wddtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> path, dist = wddtw_alignment_path(x, y)
    >>> path
    [(0, 0), (1, 1)]
    """
    cost_matrix = wddtw_cost_matrix(x, y, window, g, itakura_max_slope)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 3, y.shape[-1] - 3],
    )
