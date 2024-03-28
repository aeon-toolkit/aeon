__maintainer__ = []

from typing import Optional, Tuple, Union

import numpy as np
from numba import njit

from aeon.distances import (
    adtw_alignment_path,
    ddtw_alignment_path,
    dtw_alignment_path,
    edr_alignment_path,
    erp_alignment_path,
    msm_alignment_path,
    pairwise_distance,
    shape_dtw_alignment_path,
    squared_distance,
    twe_alignment_path,
    wddtw_alignment_path,
    wdtw_alignment_path,
)


def _medoids(
    X: np.ndarray,
    precomputed_pairwise_distance: Union[np.ndarray, None] = None,
    distance: str = "dtw",
    **kwargs,
):
    if X.shape[0] < 1:
        return X

    if precomputed_pairwise_distance is None:
        precomputed_pairwise_distance = pairwise_distance(X, metric=distance, **kwargs)

    x_size = X.shape[0]
    distance_matrix = np.zeros((x_size, x_size))
    for j in range(x_size):
        for k in range(x_size):
            distance_matrix[j, k] = precomputed_pairwise_distance[j, k]
    return X[np.argmin(sum(distance_matrix))]


def elastic_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    precomputed_medoids_pairwise_distance: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute the barycenter average of time series using a elastic distance.

    This implements an adapted version of 'petitjean' (original) DBA algorithm [1]_.

    Parameters
    ----------
    X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
            (n_cases, n_timepoints)
        A collection of time series instances to take the average from.
    distance: str or Callable, default='dtw'
        String defining the distance to use for averaging. Distance to
        compute similarity between time series. A list of valid strings for metrics
        can be found in the documentation form
        :func:`aeon.distances.get_distance_function`.
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
    center = _medoids(
        X,
        distance=distance,
        precomputed_pairwise_distance=precomputed_medoids_pairwise_distance,
        **kwargs,
    )

    cost_prev = np.inf
    if distance == "wdtw" or distance == "wddtw":
        if "g" not in kwargs:
            kwargs["g"] = 0.05
    for i in range(max_iters):
        center, cost = _ba_update(center, X, distance, **kwargs)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            break
        else:
            cost_prev = cost

        if verbose:
            print(f"[DBA aeon] epoch {i}, cost {cost}")  # noqa: T001, T201
    return center


VALID_BA_METRICS = [
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "erp",
    "edr",
    "twe",
    "msm",
    "shape_dtw",
]


@njit(cache=True, fastmath=True)
def _ba_update(
    center: np.ndarray,
    X: np.ndarray,
    distance: str = "dtw",
    window: Union[float, None] = None,
    g: float = 0.0,
    epsilon: Union[float, None] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 30,
    warp_penalty: float = 1.0,
) -> Tuple[np.ndarray, float]:
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros(X_timepoints)
    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        if distance == "dtw":
            curr_alignment, _ = dtw_alignment_path(curr_ts, center, window)
        elif distance == "ddtw":
            curr_alignment, _ = ddtw_alignment_path(curr_ts, center, window)
        elif distance == "wdtw":
            curr_alignment, _ = wdtw_alignment_path(curr_ts, center, window, g)
        elif distance == "wddtw":
            curr_alignment, _ = wddtw_alignment_path(curr_ts, center, window, g)
        elif distance == "erp":
            curr_alignment, _ = erp_alignment_path(curr_ts, center, window, g)
        elif distance == "edr":
            curr_alignment, _ = edr_alignment_path(curr_ts, center, window, epsilon)
        elif distance == "twe":
            curr_alignment, _ = twe_alignment_path(curr_ts, center, window, nu, lmbda)
        elif distance == "msm":
            curr_alignment, _ = msm_alignment_path(
                curr_ts, center, window, independent, c
            )
        elif distance == "shape_dtw":
            curr_alignment, _ = shape_dtw_alignment_path(
                curr_ts, center, window=window, descriptor=descriptor, reach=reach
            )
        elif distance == "adtw":
            curr_alignment, _ = adtw_alignment_path(
                curr_ts, center, window=window, warp_penalty=warp_penalty
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
