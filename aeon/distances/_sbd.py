"""Shape-based distance (SBD) between two time series."""

__maintainer__ = ["SebastianSchmidl"]

from typing import Optional, Union

import numpy as np
from numba import njit, objmode
from numba.typed import List as NumbaList
from scipy.signal import correlate

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    r"""
    Compute the shape-based distance (SBD) between two time series.

    Shape-based distance (SBD) [1]_ is a normalized version of
    cross-correlation (CC) that is shifting and scaling (if standardization
    is used) invariant.

    For two series, possibly of unequal length, :math:`\mathbf{x}=\{x_1,x_2,\ldots,
    x_n\}` and :math:`\mathbf{y}=\{y_1,y_2, \ldots,y_m\}`, SBD works by (optionally)
    first standardizing both time series using the z-score
    (:math:`x' = \frac{x - \mu}{\sigma}`), then computing the cross-correlation
    between x and y (:math:`CC(\mathbf{x}, \mathbf{y})`), then deviding it by the
    geometric mean of both autocorrelations of the individual sequences to normalize
    it to :math:`[-1, 1]` (coefficient normalization), and finally detecting the
    position with the maximum normalized cross-correlation:

    .. math::
        SBD(\mathbf{x}, \mathbf{y}) = 1 - max_w\left( \frac{
            CC_w(\mathbf{x}, \mathbf{y})
        }{
            \sqrt{ (\mathbf{x} \cdot \mathbf{x}) * (\mathbf{y} \cdot \mathbf{y}) }
        }\right)

    This distance method has values between 0 and 2; 0 is perfect similarity.

    The computation of the cross-correlation :math:`CC(\mathbf{x}, \mathbf{y})` for
    all values of w requires :math:`O(m^2)` time, where m is the maximum time-series
    length. We can however use the convolution theorem to our advantage, and use the
    fast (inverse) fourier transform (FFT) to perform the computation of
    :math:`CC(\mathbf{x}, \mathbf{y})` in :math:`O(m \cdot log(m))`:

    .. math::
        CC(x, y) = \mathcal{F}^{-1}\{
            \mathcal{F}(\mathbf{x}) * \mathcal{F}(\mathbf{y})
        \}

    For multivariate time series, SBD is computed independently for each channel and
    then averaged. Both time series must have the same number of channels!

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    standardize : bool, default=True
        Apply z-score to both input time series for standardization before
        computing the distance. This makes SBD scaling invariant. Default is True.

    Returns
    -------
    float
        SBD distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    See Also
    --------
    :func:`~aeon.distances.sbd_pairwise_distance` : Compute the shape-based
        distance (SBD) between all pairs of time series.

    References
    ----------
    .. [1] Paparrizos, John, and Luis Gravano: Fast and Accurate Time-Series
           Clustering. ACM Transactions on Database Systems 42, no. 2 (2017):
           8:1-8:49. https://doi.org/10.1145/3044711.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import sbd_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> dist = sbd_distance(x, y)
    """
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_sbd_distance(x, y, standardize)
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == 1 and y.shape[0] == 1:
            _x = x.ravel()
            _y = y.ravel()
            return _univariate_sbd_distance(_x, _y, standardize)
        else:
            # independent (time series should have the same number of channels!)
            nchannels = min(x.shape[0], y.shape[0])
            distance = 0.0
            for i in range(nchannels):
                distance += _univariate_sbd_distance(x[i], y[i], standardize)
            return distance / nchannels

    raise ValueError("x and y must be 1D or 2D")


def sbd_pairwise_distance(
    X: Union[np.ndarray, list[np.ndarray]],
    y: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    standardize: bool = True,
) -> np.ndarray:
    """
    Compute the shape-based distance (SBD) between all pairs of time series.

    For multivariate time series, SBD is computed independently for each channel and
    then averaged. Both time series must have the same number of channels! This is not
    checked in code for performance reasons. If the number of channels is different,
    the minimum number of channels is used.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the SBD is calculated between pairwise instances of x.
    standardize : bool, default=True
        Apply z-score to both input time series for standardization before
        computing the distance. This makes SBD scaling invariant. Default is True.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        SBD matrix between the instances of x (and y).

    Raises
    ------
    ValueError
        If x is not 2D or 3D array when only passing x.
        If x and y are not 1D, 2D or 3D arrays when passing both x and y.

    See Also
    --------
    :func:`~aeon.distances.sbd_distance` : Compute the shape-based distance between
        two time series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import sbd_pairwise_distance
    >>> # Distance between each time series in a collection of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> sbd_pairwise_distance(X)
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

    >>> # Distance between two collections of time series
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> sbd_pairwise_distance(X, y)
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> sbd_pairwise_distance(X, y_univariate)
    array([[0.],
           [0.],
           [0.]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> sbd_pairwise_distance(X)
    array([[0.        , 0.36754447, 0.5527864 ],
           [0.36754447, 0.        , 0.29289322],
           [0.5527864 , 0.29289322, 0.        ]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, _ = _convert_collection_to_numba_list(X, "", multivariate_conversion)

    if y is None:
        # To self
        return _sbd_pairwise_distance_single(_X, standardize)

    _y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
    return _sbd_pairwise_distance(_X, _y, standardize)


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance_single(
    x: NumbaList[np.ndarray], standardize: bool
) -> np.ndarray:
    n_cases = len(x)
    distances = np.zeros((n_cases, n_cases))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = sbd_distance(x[i], x[j], standardize)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance(
    x: NumbaList[np.ndarray], y: NumbaList[np.ndarray], standardize: bool
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in range(n_cases):
        for j in range(m_cases):
            distances[i, j] = sbd_distance(x[i], y[j], standardize)
    return distances


@njit(cache=True, fastmath=True)
def _univariate_sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if standardize:
        if x.size == 1 or y.size == 1:
            return 0.0

        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)

    with objmode(a="float64[:]"):
        a = correlate(x, y, method="fft")

    b = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.abs(1.0 - np.max(a / b))
