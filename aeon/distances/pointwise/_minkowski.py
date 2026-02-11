"""Minkowski distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit

from aeon.distances._distance_factory._distance_factory import (
    build_distance,
    build_pairwise_distance,
)


@njit(cache=True, fastmath=True)
def _minkowski_distance_2d(
    x: np.ndarray, y: np.ndarray, p: float, w: np.ndarray
) -> float:
    """Minkowski distance for 2D inputs with pre-validated parameters.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, shape ``(n_channels, n_timepoints)``.
    p : float
        The order of the norm (must be >= 1).
    w : np.ndarray
        Weights array of same shape as x and y.

    Returns
    -------
    float
        Minkowski distance between x and y.
    """
    dist = 0.0
    min_rows = min(x.shape[0], y.shape[0])

    for i in range(min_rows):
        min_cols = min(x[i].shape[0], y[i].shape[0])
        x_row = x[i][:min_cols]
        y_row = y[i][:min_cols]
        w_row = w[i][:min_cols]

        diff = np.abs(x_row - y_row) ** p
        dist += np.sum(w_row * diff)

    return dist ** (1.0 / p)


@njit(cache=True, fastmath=True, inline="always")
def _univariate_minkowski_distance(
    x: np.ndarray, y: np.ndarray, p: float, w: np.ndarray
) -> float:
    """Minkowski distance for univariate 1D arrays with pre-validated parameters."""
    min_length = min(x.shape[0], y.shape[0])

    x = x[:min_length]
    y = y[:min_length]
    w = w[:min_length]

    dist = np.sum(w * (np.abs(x - y) ** p))

    return float(dist ** (1.0 / p))


def _validate_minkowski_params(
    x: np.ndarray, p: float, w: np.ndarray | None
) -> np.ndarray:
    """Validate Minkowski parameters and return processed weights.

    Parameters
    ----------
    x : np.ndarray
        Input time series (for shape validation).
    p : float
        The order of the norm.
    w : np.ndarray or None
        Optional weights array.

    Returns
    -------
    np.ndarray
        Validated and processed weights array.

    Raises
    ------
    ValueError
        If p < 1 or if weights are invalid.
    """
    if p < 1:
        raise ValueError("p should be greater or equal to 1")

    if w is not None:
        _w = w.astype(x.dtype)
        if x.shape != _w.shape:
            raise ValueError("Weights w must have the same shape as x")
        if np.any(_w < 0):
            raise ValueError("Input weights should be all non-negative")
        return _w
    else:
        return np.ones_like(x)


@njit(cache=True, fastmath=True)
def _minkowski_core_wrapper(
    x: np.ndarray, y: np.ndarray, p: float, w: np.ndarray
) -> float:
    """Core Minkowski distance that handles both 1D (converted to 2D) and 2D inputs.

    This is the function passed to build_distance.
    """
    if x.ndim == 1:
        return _univariate_minkowski_distance(x, y, p, w)
    else:
        return _minkowski_distance_2d(x, y, p, w)


def minkowski_distance(
    x: np.ndarray, y: np.ndarray, p: float = 2.0, w: np.ndarray | None = None
) -> float:
    r"""Compute the Minkowski distance between two time series.

    The Minkowski distance between two time series of length m
    with a given parameter p is defined as:
    .. math::
        md(x, y, p) = \\left( \\sum_{i=1}^{n} |x_i - y_i|^p \\right)^{\\frac{1}{p}}

    Optionally, a weight vector w can be provided to
    give different weights to the elements:
    .. math::
        md_w(x, y, p, w) = \\left( \\sum_{i=1}^{n} w_i \\cdot |x_i - y_i|^p \\right)^{\\frac{1}{p}} # noqa: E501

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    p : float, default=2.0
        The order of the norm of the difference
        (default is 2.0, which represents the Euclidean distance).
    w : np.ndarray, default=None
        An array of weights, of the same shape as x and y
        (default is None, which implies equal weights).

    Returns
    -------
    float
        Minkowski distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
        If p is less than 1.
        If w is provided and its shape does not match x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import minkowski_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> minkowski_distance(x, y, p=1)
    100.0

    >>> x = np.array([1, 0, 0])
    >>> y = np.array([0, 1, 0])
    >>> w = np.array([2,2,2])
    >>> minkowski_distance(x, y, p=2, w=w) # doctest: +SKIP
    2.0
    """
    if x.ndim not in (1, 2) or y.ndim not in (1, 2):
        raise ValueError("x and y must be 1D or 2D arrays")

    _w = _validate_minkowski_params(x, p, w)
    return _minkowski_core_wrapper(x, y, p, _w)


# Build the factory version for pairwise (without validation wrapper)
_minkowski_distance_factory = build_distance(
    core_distance=_minkowski_core_wrapper,
    name="minkowski",
)


@njit(cache=True, fastmath=True)
def _minkowski_distance_no_weights(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """Minkowski distance for 2D inputs without weights.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, shape ``(n_channels, n_timepoints)``.
    p : float
        The order of the norm (must be >= 1).

    Returns
    -------
    float
        Minkowski distance between x and y.
    """
    dist = 0.0
    min_rows = min(x.shape[0], y.shape[0])

    for i in range(min_rows):
        min_cols = min(x[i].shape[0], y[i].shape[0])
        x_row = x[i][:min_cols]
        y_row = y[i][:min_cols]

        diff = np.abs(x_row - y_row) ** p
        dist += np.sum(diff)

    return dist ** (1.0 / p)


# Build the factory version for pairwise (without weights)
_minkowski_distance_factory_no_weights = build_distance(
    core_distance=_minkowski_distance_no_weights,
    name="minkowski_no_weights",
)


def minkowski_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    p: float = 2.0,
    w: np.ndarray | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute the Minkowski pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the minkoski pairwise distance between the instances of X is
        calculated.
    p : float, default=2.0
        The order of the norm of the difference
        (default is 2.0, which represents the Euclidean distance).
    w : np.ndarray, default=None
        An array of weights, applied to each pairwise calculation.
        The weights should match the shape of the time series in X and y.
        Note: weights are currently only supported for single distance computation,
        not for pairwise distances.
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. If 1, then the function is executed in a single
        thread. If greater than 1, then the function is executed in parallel.

    Returns
    -------
    np.ndarray
        Minkowski pairwise distance matrix between
        the instances of X (and y if provided).

    Raises
    ------
    ValueError
        If X (and y if provided) are not 1D, 2D or 3D arrays.
        If p is less than 1.
        If w is provided and its shape does not match the instances in X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import minkowski_pairwise_distance
    >>> X = np.array([[[1, 2, 3, 4]],[[4, 5, 6, 3]], [[7, 8, 9, 3]]])
    >>> minkowski_pairwise_distance(X, p=1)
    array([[ 0., 10., 19.],
           [10.,  0.,  9.],
           [19.,  9.,  0.]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y = np.array([[[11, 12, 13]],[[14, 15, 16]], [[17, 18, 19]]])
    >>> minkowski_pairwise_distance(X, y,p=2)
    array([[17.32050808, 22.5166605 , 27.71281292],
           [12.12435565, 17.32050808, 22.5166605 ],
           [ 6.92820323, 12.12435565, 17.32050808]])

    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> y_univariate = np.array([11, 12, 13])
    >>> minkowski_pairwise_distance(X, y_univariate, p=1)
    array([[30.],
           [21.],
           [12.]])

    >>> # Distance between each TS in a collection of unequal-length time series
    >>> X = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11, 12])]
    >>> minkowski_pairwise_distance(X)
    array([[ 0.        ,  5.19615242, 12.12435565],
           [ 5.19615242,  0.        ,  8.        ],
           [12.12435565,  8.        ,  0.        ]])
    """
    # Validate p parameter
    if p < 1:
        raise ValueError("p should be greater or equal to 1")

    # For now, weights are not supported in pairwise mode (to be implemented)
    if w is not None:
        raise NotImplementedError(
            "Weights are not currently supported for minkowski_pairwise_distance. "
            "Use minkowski_distance for weighted distance computation."
        )

    # Build the pairwise function dynamically with the current p
    pairwise_func = build_pairwise_distance(
        core_distance=_minkowski_distance_factory_no_weights,
        name="minkowski",
    )

    return pairwise_func(X, y, n_jobs, p)
