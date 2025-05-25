from typing import Optional, Union

import numpy as np
from numba import njit

from aeon.clustering.averaging._ba_utils import (
    _get_alignment_path,
    _get_init_barycenter,
)


def petitjean_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    init_barycenter: Union[np.ndarray, str] = "mean",
    weights: Optional[np.ndarray] = None,
    precomputed_medoids_pairwise_distance: Optional[np.ndarray] = None,
    verbose: bool = False,
    random_state: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Compute the barycenter average of time series using a elastic distance.

    This implements an adapted version of 'petitjean' (original) DBA algorithm [1]_.

    Parameters
    ----------
    X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
            (n_cases, n_timepoints)
        A collection of time series instances to take the average from.
    distance: str, default='dtw'
        String defining the distance to use for averaging. Distance to
        compute similarity between time series. A list of valid strings for metrics
        can be found in the documentation form
        :func:`aeon.distances.get_distance_function`.
    max_iters: int, default=30
        Maximum number iterations for dba to update over.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the Expectation-Maximization procedure stops.
    init_barycenter: np.ndarray or, default=None
        The initial barycenter to use for the minimisation. If a np.ndarray is provided
        it must be of shape ``(n_channels, n_timepoints)``. If a str is provided it must
        be one of the following: ['mean', 'medoids', 'random'].
    weights: Optional[np.ndarray] of shape (n_cases,), default=None
        The weights associated to each time series instance, if None a weight
        of 1 will be associated to each instance.
    precomputed_medoids_pairwise_distance: np.ndarray (of shape (len(X), len(X)),
                default=None
        Precomputed medoids pairwise.
    verbose: bool, default=False
        Boolean that controls the verbosity.
    random_state: int or None, default=None
        Random state to use for the barycenter averaging.
    **kwargs
        Keyword arguments to pass to the distance method.

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

    if X.ndim == 3:
        _X = X
    elif X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        raise ValueError("X must be a 2D or 3D array")

    if weights is None:
        weights = np.ones(len(_X))

    barycenter = _get_init_barycenter(
        _X,
        init_barycenter,
        distance,
        precomputed_medoids_pairwise_distance,
        random_state,
        **kwargs,
    )

    cost_prev = np.inf
    if distance == "wdtw" or distance == "wddtw":
        if "g" not in kwargs:
            kwargs["g"] = 0.05
    for i in range(max_iters):
        barycenter, cost = _ba_one_iter_petitjean(
            barycenter, _X, distance, weights, **kwargs
        )
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            break
        else:
            cost_prev = cost

        if verbose:
            print(f"[DBA] epoch {i}, cost {cost}")  # noqa: T001, T201
    return barycenter


@njit(cache=True, fastmath=True)
def _ba_one_iter_petitjean(
    barycenter: np.ndarray,
    X: np.ndarray,
    distance: str = "dtw",
    weights: Optional[np.ndarray] = None,
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
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros(X_timepoints)
    alignment = np.zeros((X_dims, X_timepoints))
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, curr_cost = _get_alignment_path(
            barycenter,
            curr_ts,
            distance,
            window,
            g,
            epsilon,
            nu,
            lmbda,
            independent,
            c,
            descriptor,
            reach,
            warp_penalty,
            transformation_precomputed,
            transformed_x,
            transformed_y,
        )

        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j] * weights[i]
            sum[k] += 1 * weights[i]
        cost += curr_cost * weights[i]

    return alignment / sum, cost
