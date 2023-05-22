# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import Callable, Tuple

import numpy as np
from numba import njit

from aeon.clustering.metrics.medoids import medoids
from aeon.distances import (
    ddtw_alignment_path,
    dtw_alignment_path,
    edr_alignment_path,
    erp_alignment_path,
    msm_alignment_path,
    squared_distance,
    twe_alignment_path,
    wddtw_alignment_path,
    wdtw_alignment_path,
)


def dba(
    X: np.ndarray,
    metric: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    medoids_distance_metric: str = "dtw",
    precomputed_medoids_pairwise_distance: np.ndarray = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute the dtw barycenter average of time series.

    This implements the 'petitjean' version (original) DBA algorithm [1]_.


    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints)
        A collection of time series instances to take the average from.
    metric: str or Callable, defaults = 'dtw'
        String that is the distance metric to use for averaging.
        If Callable provided must be of the form (x, y) -> (float, np.ndarray)
        where the first element is the distance and the second is the alignment path.
    max_iters: int, defaults = 30
        Maximum number iterations for dba to update over.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the Expectation-Maximization procedure stops.
    medoids_distance_metric: str, defaults = 'euclidean'
        String that is the distance metric to use with medoids
    precomputed_medoids_pairwise_distance: np.ndarray (of shape (len(X), len(X)),
                defulats = None
        Precomputed medoids pairwise.
    verbose: bool, defaults = False
        Boolean that controls the verbosity.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the average of the collaction of instances provided.

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    if len(X) <= 1:
        return X

    # center = X.mean(axis=0)
    center = medoids(
        X,
        distance_metric=medoids_distance_metric,
        precomputed_pairwise_distance=precomputed_medoids_pairwise_distance,
        **kwargs,
    )

    cost_prev = np.inf
    for i in range(max_iters):
        if metric == "dtw":
            center, cost = _dtw_dba_update(
                center,
                X,
                kwargs.get("window"),
            )
        elif metric == "ddtw":
            center, cost = _ddtw_dba_update(
                center,
                X,
                kwargs.get("window"),
            )
        elif metric == "wdtw":
            center, cost = _wdtw_dba_update(
                center,
                X,
                kwargs.get("window"),
                kwargs.get("g", 0.05),
            )
        elif metric == "wddtw":
            center, cost = _wddtw_dba_update(
                center,
                X,
                kwargs.get("window"),
                kwargs.get("g", 0.05),
            )
        elif metric == "erp":
            center, cost = _erp_dba_update(
                center,
                X,
                kwargs.get("window"),
                kwargs.get("g", 0.0),
            )
        elif metric == "edr":
            center, cost = _edr_dba_update(
                center,
                X,
                kwargs.get("window"),
                kwargs.get("epsilon"),
            )
        elif metric == "twe":
            center, cost = _twe_dba_update(
                center,
                X,
                kwargs.get("window"),
                kwargs.get("nu", 0.001),
                kwargs.get("lmbda", 1.0),
            )
        elif metric == "msm":
            center, cost = _msm_dba_update(
                center,
                X,
                kwargs.get("window"),
                kwargs.get("independent", True),
                kwargs.get("c", 1.0),
            )
        else:
            if isinstance(metric, Callable):
                center, cost = _dba_update(center, X, metric)
            else:
                raise ValueError(
                    f"Metric must be a known string or Callable, got {metric}"
                )

        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            break
        else:
            cost_prev = cost

        if verbose:
            print(f"[DBA aeon] epoch {i}, cost {cost}")  # noqa: T001, T201
    return center


@njit(cache=True, fastmath=True)
def _dba_update(
    center: np.ndarray, X: np.ndarray, path_callable: Callable
) -> Tuple[np.ndarray, float]:
    """Perform an update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    path_callable: Callable[Union[np.ndarray, np.ndarray], tuple[list[tuple], float]]
        Callable that returns the distance path.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = path_callable(curr_ts, center)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += np.linalg.norm(curr_ts[:, j], center[:, k]) ** 2

    return alignment / sum, cost / X_timepoints


@njit(cache=True, fastmath=True)
def _dtw_dba_update(
    center: np.ndarray, X: np.ndarray, window: float = None
) -> Tuple[np.ndarray, float]:
    """Perform a dtw update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = dtw_alignment_path(curr_ts, center, window)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints


@njit(cache=True, fastmath=True)
def _ddtw_dba_update(
    center: np.ndarray, X: np.ndarray, window: float = None
) -> Tuple[np.ndarray, float]:
    """Perform a ddtw update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = ddtw_alignment_path(curr_ts, center, window)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints


@njit(cache=True, fastmath=True)
def _wdtw_dba_update(
    center: np.ndarray, X: np.ndarray, window: float = None, g: float = 0.05
) -> Tuple[np.ndarray, float]:
    """Perform a wdtw update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    window: float, default=0.05
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = wdtw_alignment_path(curr_ts, center, window, g)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints


@njit(cache=True, fastmath=True)
def _wddtw_dba_update(
    center: np.ndarray, X: np.ndarray, window: float = None, g: float = 0.05
) -> Tuple[np.ndarray, float]:
    """Perform a wddtw update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float, defaults=0.05
        Constant that controls the level of penalisation for the points with larger
        phase difference.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = wddtw_alignment_path(curr_ts, center, window, g)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints


@njit(cache=True, fastmath=True)
def _erp_dba_update(
    center: np.ndarray, X: np.ndarray, window: float = None, g: float = 0.0
) -> Tuple[np.ndarray, float]:
    """Perform a wddtw update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    g: float or np.ndarray of shape (n_channels), defaults=0.
        The reference value to penalise gaps. The default is 0. If it is an array
        then it must be the length of the number of channels in x and y. If a single
        value is provided then that value is used across each channel.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = erp_alignment_path(curr_ts, center, window, g)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints


@njit(cache=True, fastmath=True)
def _edr_dba_update(
    center: np.ndarray, X: np.ndarray, window: float = None, epsilon: float = None
) -> Tuple[np.ndarray, float]:
    """Perform a wddtw update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, defaults = None
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. If not specified as per the original paper
        epsilon is set to a quarter of the maximum standard deviation.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = edr_alignment_path(curr_ts, center, window, epsilon)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints


@njit(cache=True, fastmath=True)
def _twe_dba_update(
    center: np.ndarray,
    X: np.ndarray,
    window: float = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Perform a wddtw update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    nu: float, defaults = 0.001
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    lmbda: float, defaults = 1.0
        A constant penalty that punishes the editing efforts. Must be >= 1.0.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = twe_alignment_path(curr_ts, center, window, nu, lmbda)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints


@njit(cache=True, fastmath=True)
def _msm_dba_update(
    center: np.ndarray,
    X: np.ndarray,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Perform an update iteration for dba.

    Parameters
    ----------
    center: np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current center (or average).
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints)
        Time series instances compute average from.
    window: float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    independent: bool, defaults=True
        Whether to use the independent or dependent MSM distance. The
        default is True (to use independent).
    c: float, defaults=1.
        Cost for split or merge operation. Default is 1.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the current iteration center (or average).
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = msm_alignment_path(curr_ts, center, window, independent, c)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints
