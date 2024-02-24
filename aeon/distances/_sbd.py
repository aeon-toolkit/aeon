"""Shape-based distance (SBD) distance between two time series."""

__author__ = ["codelionx"]

from typing import Optional

import numpy as np
from numba import njit, objmode
from scipy.signal import correlate, fftconvolve

from aeon.distances._utils import reshape_pairwise_to_multiple


@njit(cache=True, fastmath=True)
def sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    r"""Compute the shape-based distance (SBD) between two time series.

    Shape-based distance (SBD) [1]_ is a normalized version of
    cross-correlation (CC) that is shifting and scaling (if standardization
    is used) invariant.

    For two series, possibly of unequal length, :math:`\mathbf{x}=\{x_1,x_2,\ldots,
    x_n\}` and :math:`\mathbf{y}=\{y_1,y_2, \ldots,y_m\}`, BSD works by (optionally)
    first standardizing both time series using the z-score
    (:math:`x' = \frac{x - \mu}{\sigma}`), then computing the cross-correlation
    between x and y (:math:`CC(\mathbf{x}, \mathbf{y})`), then deviding it by the
    geometric mean of both autocorrelations of the individual sequences to normalize
    it to :math:`[-1 1]` (coefficient normalization), and finally detecting the
    position with the maximum normalized cross-correlation:

    .. math:
        SBD(\mathbf{x}, \mathbf{y}) = 1 - max_w( \frac{
            CC_w(\mathbf{x}, \mathbf{y})
        }{
            \sqrt{ \mathbf{x} \cdot \mathbf{x} * \mathbf{y} \cdot \mathbf{y} }
        })

    This distance measure has values between 0 and 2; 0 is perfect similarity.

    The computation of the cross-correlation :math:`CC(\mathbf{x}, \mathbf{y})` for
    all values of w requires :math:`O(m^2)` time, where m is the maximum time-series
    length.
    We can however use the convolution theorem to our advantage, and use the fast
    (inverse) fourier transform (FFT) to perform the computation of CC(x, y) in
    :math:`O(m log(m))`:

    .. math:
        CC(x, y) = \mathcal{F}^{-1}\{\mathcal{F}(\mathbf{x}) * \mathcal{F}(\mathbf{y})}.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    standardize : bool
        Apply z-score to both input time series for standardization before
        computing the distance. This make SBD scaling invariant.

    Returns
    -------
    float
        BSD distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

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
        return _univariate_sbd_distance(x, y)
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == y.shape[0] == 1:
            _x = x.ravel()
            _y = y.ravel()
            return _univariate_sbd_distance(_x, _y)
        else:
            # independent
            nchannels = min(x.shape[0], y.shape[0])
            distance = 0.0
            for i in range(nchannels):
                distance += _univariate_sbd_distance(x[i], y[i])
            return distance

    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def sbd_pairwise_distance(x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    if y is None:
        # To self
        if x.ndim == 3:
            return _sbd_pairwise_distance_single(x)
        elif x.ndim == 2:
            _X = x.reshape((x.shape[0], 1, x.shape[1]))
            return _sbd_pairwise_distance_single(_X)
        raise ValueError("X must be 2D or 3D")

    _x, _y = reshape_pairwise_to_multiple(x, y)
    return _sbd_pairwise_distance(_x, _y)


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance_single(x: np.ndarray) -> np.ndarray:
    n_instances = x.shape[0]
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = sbd_distance(x[i], x[j])
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = sbd_distance(x[i], y[j])
    return distances


@njit(cache=True, fastmath=True)
def _univariate_sbd_distance(x: np.ndarray, y: np.ndarray) -> float:
    # option 1:
    with objmode(a="float64[:]"):
        a = correlate(x, y, method="fft")
    # option 2:
    # a = np.convolve(x, _reverse_and_conj(y))

    b = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.abs(1.0 - np.max(a / b))


@njit(cache=True, fastmath=True)
def _reverse_and_conj(x: np.ndarray) -> np.ndarray:
    """Reverse array `x` in all dimensions and perform the complex conjugate."""
    if x.ndim == 1:
        return x[::-1].conj()
    elif x.ndim == 2:
        return x[::-1, ::-1].conj()
    elif x.ndim == 3:
        return x[::-1, ::-1, ::-1].conj()
    else:
        raise NotImplementedError(
            "reverse_and_conj ist not implemented for arrays of "
            "dimensionality 4 or larger!"
        )
    # Implementation in scipy (not compatible with Numba):
    # reverse = (slice(None, None, -1),) * x.ndim
    # return x[reverse].conj()
