# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import Tuple

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


def elastic_barycenter_average(
    X: np.ndarray,
    metric: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    precomputed_medoids_pairwise_distance: np.ndarray = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute the barycenter average of time series using a elastic distance.

    This implements an adapted version of 'petitjean' (original) DBA algorithm [1]_.

    Parameters
    ----------
    X: np.ndarray, of shape (n_instances, n_channels, n_timepoints) or
            (n_instances, n_timepoints)
        A collection of time series instances to take the average from.
    metric: str or Callable, default='dtw'
        String that is the distance metric to use for averaging.
        If Callable provided must be of the form (x, y) -> (float, np.ndarray)
        where the first element is the distance and the second is the alignment path.
    max_iters: int, default=30
        Maximum number iterations for dba to update over.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the Expectation-Maximization procedure stops.
    precomputed_medoids_pairwise_distance: np.ndarray (of shape (len(X), len(X)),
                default=None
        Precomputed medoids pairwise.
    verbose: bool, default=False
        Boolean that controls the verbosity.
    **kwargs
        Keyword arguments to pass to the distance metric.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the average of the collection of instances provided.

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
        distance_metric=metric,
        precomputed_pairwise_distance=precomputed_medoids_pairwise_distance,
        **kwargs,
    )

    cost_prev = np.inf
    if metric == "wdtw" or metric == "wddtw":
        if "g" not in kwargs:
            kwargs["g"] = 0.05
    for i in range(max_iters):
        center, cost = _ba_update(center, X, metric, **kwargs)
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
def _ba_update(
    center: np.ndarray,
    X: np.ndarray,
    metric: str = "dtw",
    window: float = None,
    g: float = 0.0,
    epsilon: float = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
) -> Tuple[np.ndarray, float]:
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros(X_timepoints)
    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        if metric == "dtw":
            curr_alignment, _ = dtw_alignment_path(curr_ts, center, window)
        elif metric == "ddtw":
            curr_alignment, _ = ddtw_alignment_path(curr_ts, center, window)
        elif metric == "wdtw":
            curr_alignment, _ = wdtw_alignment_path(curr_ts, center, window, g)
        elif metric == "wddtw":
            curr_alignment, _ = wddtw_alignment_path(curr_ts, center, window, g)
        elif metric == "erp":
            curr_alignment, _ = erp_alignment_path(curr_ts, center, window, g)
        elif metric == "edr":
            curr_alignment, _ = edr_alignment_path(curr_ts, center, window, epsilon)
        elif metric == "twe":
            curr_alignment, _ = twe_alignment_path(curr_ts, center, window, nu, lmbda)
        elif metric == "msm":
            curr_alignment, _ = msm_alignment_path(
                curr_ts, center, window, independent, c
            )
        else:
            # When numba version > 0.57 add more informative error with what metric
            # was passed.
            raise ValueError("Metric parameter invalid")
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += squared_distance(curr_ts[:, j], center[:, k])

    return alignment / sum, cost / X_timepoints
