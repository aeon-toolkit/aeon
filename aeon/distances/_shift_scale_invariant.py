"""Shift-invariant distance."""

from typing import Optional, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def shift_scale_invariant_distance(
    x: np.ndarray, y: np.ndarray, max_shift: Optional[int] = None
) -> float:
    r"""Compute the shift and scale invariant distance [1]_ between two time series.

    The shift and scale invariant distance is designed to compare two time series while
    being robust to both shifts in time (translation) and changes in scale. The method
    uses an optimisation process to find the best shift (``q``) and scale factor
    (``α``) that minimises the distance between the two time series.

    Given two time series, the second time series is shifted by a maximum of
    ``max_shift`` time points to the left or right, and for each shift, a scaling
    factor is computed that minimises the L2 norm between the scaled and shifted
    version of the second time series and the first time series.

    The optimal scale factor ``α`` is computed for a fixed shift ``q`` using a
    closed-form solution derived from minimising the distance. The algorithm starts by
    aligning the peaks of the two time series, and then adjusts the shift locally to
    find the optimal ``q``. The shift and scale that yield the smallest distance are
    returned as the final distance.

    Formally the shift and scale invariant distance is defined as:

    .. math::
        \hat{d}(a, b) = \min_{\alpha, q} \frac{L_2(a, \alpha b_{(q)})}{L_2(a, a)}

    where alpha is defined as:

    .. math::
        \alpha = \frac{L_2(a, b_{(q)})^2}{L_2(b_{(q)},b_{(q)})^2}


    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    max_shift : int or None, default=None
        Maximum shift allowed in the alignment path. If None, then max_shift is set
        to min(x.shape[-1], y.shape[-1]).

    Returns
    -------
    float
        Shift-scale invariant distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] J. Yang and J. Leskovec. Patterns of temporal variation in online media. In
    Proc. of the fourth ACM international conf. on Web search and data mining, page
    177. ACM, 2011.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import shift_scale_invariant_distance
    >>> x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    >>> y = np.array([11., 12., 13., 14., 15., 16., 17., 18., 19., 20.])
    >>> univ_dist = shift_scale_invariant_distance(x, y)

    >>> x = np.array([[1., 2., 3., 4.], [6., 7., 8., 9.], [0., 1., 0., 2.]])
    >>> y = np.array([[11., 12., 13., 14.], [3., 22., 5., 4.], [12., 3., 4., 5.]])
    >>> multi_dist = shift_scale_invariant_distance(x, y)
    """
    if max_shift is None:
        max_shift = min(x.shape[-1], y.shape[-1])

    if x.ndim == 1 and y.ndim == 1:
        dist, _ = _univariate_shift_scale_invariant_distance(x, y, max_shift)
        return dist
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == 1 and y.shape[0] == 1:
            dist, _ = _univariate_shift_scale_invariant_distance(
                x[0, :], y[0, :], max_shift
            )
            return dist
        else:
            n_channels = min(x.shape[0], y.shape[0])
            distance = 0.0
            for i in range(n_channels):
                dist, _ = _univariate_shift_scale_invariant_distance(
                    x[i], y[i], max_shift
                )
                distance += dist

            return distance
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _scale_d(x: np.ndarray, y: np.ndarray) -> float:
    denominator = np.dot(y, y)

    # Prevent divide by zero
    if denominator == 0:
        return float(np.finfo(np.float64).max)

    alpha = np.dot(x, y) / denominator

    norm_x = np.linalg.norm(x)
    # Prevent divide by zero
    if norm_x == 0:
        return float(np.finfo(np.float64).max)

    dist = np.linalg.norm(x - alpha * y) / norm_x

    return dist


@njit(cache=True, fastmath=True)
def _univariate_shift_scale_invariant_distance(
    x: np.ndarray, y: np.ndarray, max_shift: int
) -> tuple[float, np.ndarray]:
    if np.array_equal(x, y):
        return 0.0, y
    min_dist = _scale_d(x, y)
    best_shifted_y = y

    for sh in range(-max_shift, max_shift + 1):
        if sh == 0:
            shifted_y = y
        elif sh < 0:
            # Shift left
            shifted_y = np.append(y[-sh:], np.zeros(-sh))
        else:
            # Shift right
            shifted_y = np.append(np.zeros(sh), y[:-sh])

        dist = _scale_d(x, shifted_y)

        if dist < min_dist:
            min_dist = dist
            best_shifted_y = shifted_y

    return min_dist, best_shifted_y


def shift_scale_invariant_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    max_shift: Optional[int] = None,
) -> np.ndarray:
    r"""Compute the shift-scale invariant pairwise distance between time series.

    By default, this takes a collection of :math:`n` time series :math:`X` and returns a
    matrix
    :math:`D` where :math:`D_{i,j}` is the shift-scale distance between the
    :math:`i^{th}` and the :math:`j^{th}` series in :math:`X`. If :math:`X` is 2
    dimensional, it is assumed to be a collection of univariate series with shape
    ``(n_cases, n_timepoints)``. If it is 3 dimensional, it is assumed to be shape
    ``(n_cases, n_channels, n_timepoints)``.

    This function has an optional argument, :math:`y`, to allow calculation of the
    distance matrix between :math:`X` and one or more series stored in :math:`y`. If
    :math:`y` is 1 dimensional, we assume it is a single univariate series and the
    distance matrix returned is shape ``(n_cases,1)``. If it is 2D, we assume it
    is a collection of univariate series with shape ``(m_cases, m_timepoints)``
    and the distance ``(n_cases,m_cases)``. If it is 3 dimensional,
    it is assumed to be shape ``(m_cases, m_channels, m_timepoints)``.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the dtw pairwise distance between the instances of X is
        calculated.
    max_shift : int or None, default=None
        Maximum shift allowed in the alignment path. If None, then max_shift is set
        to min(X.shape[-1], y.shape[-1]) or if y is None, max_shift is set to
        X.shape[-1].

    Returns
    -------
    np.ndarray
        Shift-scale invariant  pairwise matrix between the instances of X of shape
        ``(n_cases, n_cases)`` or between X and y of shape ``(n_cases,
        n_cases)``.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array and if y is not 1D, 2D or 3D arrays when passing y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import shift_scale_invariant_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1., 2., 3.]],[[4., 5., 6.]], [[7., 8., 9.]]])
    >>> pw_self = shift_scale_invariant_pairwise_distance(X)

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1., 2., 3.]],[[4., 5., 6.]], [[7., 8., 9.]]])
    >>> y = np.array([[[11., 12., 13.]],[[14., 15., 16.]], [[17., 18., 19.]]])
    >>> pw = shift_scale_invariant_pairwise_distance(X, y)

    >>> X = np.array([[[1., 2., 3.]],[[4., 5., 6.]], [[7., 8., 9.]]])
    >>> y_univariate = np.array([11., 12., 13.])
    >>> single_pw =shift_scale_invariant_pairwise_distance(X, y_univariate)
    """
    if max_shift is None:
        if y is None:
            max_shift = X.shape[-1]
        else:
            max_shift = min(X.shape[-1], y.shape[-1])
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, _ = _convert_collection_to_numba_list(X, "", multivariate_conversion)

    if y is None:
        return _shift_invariant_pairwise_distance(_X, _X, max_shift)

    _y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
    return _shift_invariant_pairwise_distance(_X, _y, max_shift)


def shift_scale_invariant_best_shift(
    x: np.ndarray, y: np.ndarray, max_shift: int = None
) -> tuple[float, np.ndarray]:
    """Compute the best shift for the shift-invariant distance.

    This function computes the best shift for the shift-invariant distance between
    two time series. The best shift value returned is the shift of y that minimises the
    distance between x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    max_shift : int or None, default=None
        Maximum shift allowed in the alignment path. If None, then max_shift is set
        to min(x.shape[1], y.shape[1]).

    Returns
    -------
    float
        Shift-scale invariant distance between x and y, minimum value 0.
    np.ndarray
        Shifts of y that minimise the distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import shift_scale_invariant_best_shift
    >>> x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    >>> y = np.array([11., 12., 13., 14., 15., 16., 17., 18., 19., 20.])
    >>> univ_dist, univ_shift = shift_scale_invariant_best_shift(x, y)

    >>> x = np.array([[1., 2., 3., 4.], [6., 7., 8., 9.], [0., 1., 0., 2.]])
    >>> y = np.array([[11., 12., 13., 14.],[7., 8., 9., 20.],[1., 3., 4., 5.]])
    >>> multi_dist, multi_shift = shift_scale_invariant_best_shift(x, y)
    """
    if max_shift is None:
        max_shift = min(x.shape[-1], y.shape[-1])
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_shift_scale_invariant_distance(x, y, max_shift)
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == 1 and y.shape[0] == 1:
            return _univariate_shift_scale_invariant_distance(
                x[0, :], y[0, :], max_shift
            )
        else:
            n_channels = min(x.shape[0], y.shape[0])
            distance = 0.0
            best_shift = np.zeros((n_channels, y.shape[1]))
            for i in range(n_channels):
                dist, curr_shift = _univariate_shift_scale_invariant_distance(
                    x[i], y[i], max_shift
                )
                best_shift[i] = curr_shift
                distance += dist

            return distance, best_shift

    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _shift_invariant_pairwise_distance(
    x: NumbaList[np.ndarray], y: NumbaList[np.ndarray], max_shift: int
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = shift_scale_invariant_distance(x[i], y[j], max_shift)
    return distances
