"""Shape-based distance (SBD) between two time series."""

__maintainer__ = ["SebastianSchmidl"]

from typing import List

import numpy as np
from numba import njit, objmode, prange
from numba.typed import List as NumbaList
from scipy.signal import correlate

from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.numba._threading import threaded
from aeon.utils.validation.collection import _is_numpy_list_multivariate

__all__ = [
    "sbd_distance",
    "sbd_pairwise_distance",
    "_univariate_sbd_distance",
    "_univariate_sbd_ncc_curve",
    "_univariate_sbd_best_shift",
    "_univariate_sbd_align_to_center",
    "_zscore_1d",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    between x and y (:math:`CC(\mathbf{x}, \mathbf{y})`), then dividing it by the
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


@threaded
def sbd_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    standardize: bool = True,
    n_jobs: int = 1,
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
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. If 1, then the function is executed in a single
        thread. If greater than 1, then the function is executed in parallel.

        NOTE: For this distance function unless your data has a large number of time
        points, it is recommended to use n_jobs=1.

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
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, _ = _convert_collection_to_numba_list(X, "", multivariate_conversion)

    if y is None:
        # To self
        return _sbd_pairwise_distance_single(_X, standardize)

    _y, _ = _convert_collection_to_numba_list(y, "y", multivariate_conversion)
    return _sbd_pairwise_distance(_X, _y, standardize)


# ---------------------------------------------------------------------------
# Pairwise internals
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, parallel=True)
def _sbd_pairwise_distance_single(
    x: NumbaList[np.ndarray], standardize: bool
) -> np.ndarray:
    n_cases = len(x)
    distances = np.zeros((n_cases, n_cases))

    for i in prange(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = sbd_distance(x[i], x[j], standardize)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _sbd_pairwise_distance(
    x: NumbaList[np.ndarray], y: NumbaList[np.ndarray], standardize: bool
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    for i in prange(n_cases):
        for j in range(m_cases):
            distances[i, j] = sbd_distance(x[i], y[j], standardize)
    return distances


# ---------------------------------------------------------------------------
# Core reusable pieces: z-score, NCC curve, best shift, alignment, distance
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _zscore_1d(x: np.ndarray) -> np.ndarray:
    """Numba-friendly z-score of a 1D array.

    If the variance is zero, returns all zeros.
    """
    x = x.astype(np.float64)
    n = x.size
    if n <= 1:
        return x * 0.0

    mu = 0.0
    for i in range(n):
        mu += x[i]
    mu /= n

    var = 0.0
    for i in range(n):
        diff = x[i] - mu
        var += diff * diff
    var /= n

    if var <= 0.0:
        return x * 0.0

    sigma = np.sqrt(var)
    out = np.empty_like(x)
    for i in range(n):
        out[i] = (x[i] - mu) / sigma
    return out


@njit(cache=True, fastmath=True)
def _univariate_sbd_preprocess(
    x: np.ndarray, y: np.ndarray, standardize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Cast to float64 and optionally standardize both series."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if standardize:
        if x.size == 1 or y.size == 1:
            # Degenerate case: we define distance 0.0 later, but still return arrays.
            return x * 0.0, y * 0.0

        x = _zscore_1d(x)
        y = _zscore_1d(y)

    return x, y


@njit(cache=True, fastmath=True)
def _univariate_sbd_ncc_curve(
    x: np.ndarray, y: np.ndarray, standardize: bool
) -> np.ndarray:
    """Return the normalized cross-correlation (NCC) curve for all lags.

    Uses scipy.signal.correlate via objmode + FFT method, then normalizes
    by the geometric mean of autocorrelations, exactly as in the original
    SBD definition.
    """
    x, y = _univariate_sbd_preprocess(x, y, standardize)

    # full cross-correlation via SciPy in objmode
    with objmode(a="float64[:]"):
        a = correlate(x, y, method="fft")

    # coefficient normalisation
    b = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if b == 0.0:
        # avoid division by zero; all zeros → NCC all zeros
        return a * 0.0

    return a / b


@njit(cache=True, fastmath=True)
def _univariate_sbd_best_shift(
    center: np.ndarray, x: np.ndarray, standardize: bool
) -> int:
    """Return the lag that maximises NCC(center, x).

    Using SciPy's correlate semantics:

    correlate(center, x, "full") yields lags from
    -(len(x)-1) to +(len(center)-1); zero-lag is at index len(x)-1.
    """
    ncc = _univariate_sbd_ncc_curve(center, x, standardize)
    idx = np.argmax(ncc)

    # For equal-length series (n = len(center) = len(x)):
    #   lags = -(n-1), ..., 0, ..., +(n-1)
    #   zero-lag at index (n-1)
    # In general:
    shift = idx - (x.size - 1)
    return shift


@njit(cache=True, fastmath=True)
def _roll_zeropad_1d(a: np.ndarray, shift: int) -> np.ndarray:
    """Roll a 1D array by 'shift' with zeros padding in the gaps.

    Positive shift => move data to the right.
    Negative shift => move data to the left.
    """
    n = a.size
    res = np.zeros_like(a)

    for i in range(n):
        j = i - shift
        if 0 <= j < n:
            res[i] = a[j]
    return res


@njit(cache=True, fastmath=True)
def _univariate_sbd_align_to_center(
    center: np.ndarray, x: np.ndarray, standardize: bool
) -> np.ndarray:
    """Align x to center using SBD (max NCC) and return shifted x.

    If center is all zeros, x is returned unchanged.
    """
    if np.all(center == 0.0):
        return x.copy()
    shift = _univariate_sbd_best_shift(center, x, standardize)
    return _roll_zeropad_1d(x, shift)


@njit(cache=True, fastmath=True)
def _univariate_sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool) -> float:
    """Core univariate SBD distance using the NCC helper."""
    ncc = _univariate_sbd_ncc_curve(x, y, standardize)
    return np.abs(1.0 - np.max(ncc))
