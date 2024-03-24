from typing import Optional, Union

import numpy as np
from numba import njit
from sklearn.utils import check_random_state

from aeon.clustering.averaging._ba_utils import (
    _get_alignment_path,
    _get_init_barycenter,
)


def subgradient_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol=1e-5,
    init_barycenter: Union[np.ndarray, str] = "mean",
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    precomputed_medoids_pairwise_distance: Optional[np.ndarray] = None,
    verbose: bool = False,
    random_state: int = 1,
    **kwargs,
) -> np.ndarray:
    """Compute the stochastic subgradient barycenter average of time series.

    Stochastic subgradient is much faster than petitjean, however, it is not guaranteed
    to find the optimal solution.

    This implements a stochastic subgradient DBA algorithm. This changes how
    the average is computed. Unlike traditional methods, it calculates a subgradient
    based on each individual time series within the dataset. The barycenter is then
    iteratively updated using these subgradient. See [2]_ for more details.

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
        be one of the following: ['mean', 'medoids'].
    initial_step_size : float (default: 0.05)
        Initial step size for the subgradient descent algorithm.
        Default value is suggested by [2]_.
    final_step_size : float (default: 0.005)
        Final step size for the subgradient descent algorithm.
        Default value is suggested by [2]_.
    precomputed_medoids_pairwise_distance: np.ndarray (of shape (len(X), len(X)),
                default=None
        Precomputed medoids pairwise.
    verbose: bool, default=False
        Boolean that controls the verbosity.
    random_state: int, default=1
        Random state to use for the barycenter averaging.
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
    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    if len(X) <= 1:
        return X

    if X.ndim == 3:
        _X = X
    elif X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        raise ValueError("X must be a 2D or 3D array")

    barycenter = _get_init_barycenter(
        _X,
        init_barycenter,
        distance,
        precomputed_medoids_pairwise_distance,
        random_state,
        **kwargs,
    )

    random_state = check_random_state(random_state)

    cost_prev = np.inf
    if distance == "wdtw" or distance == "wddtw":
        if "g" not in kwargs:
            kwargs["g"] = 0.05

    eta = initial_step_size
    X_size = _X.shape[0]
    for i in range(max_iters):
        shuffled_indices = random_state.permutation(X_size)
        barycenter, cost, eta = _ba_one_iter_subgradient(
            barycenter,
            _X,
            shuffled_indices,
            distance,
            initial_step_size,
            final_step_size,
            eta,
            i,
            **kwargs,
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
def _ba_one_iter_subgradient(
    barycenter: np.ndarray,
    X: np.ndarray,
    shuffled_indices: np.ndarray,
    distance: str = "dtw",
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    eta: float = 0.05,
    iteration: int = 0,
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
):

    X_size, X_dims, X_timepoints = X.shape
    cost = 0.0
    # Only update eta on the first iteration
    eta_decrement = 0.0
    if iteration == 0:
        eta_decrement = (initial_step_size - final_step_size) / X_size

    for i in shuffled_indices:
        curr_ts = X[i]
        curr_alignment, curr_cost = _get_alignment_path(
            barycenter,
            X[i],
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
        )

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter[:, k] - curr_ts[:, j]

        barycenter -= (2.0 * eta) * new_ba

        eta -= eta_decrement
        cost = curr_cost
    return barycenter, cost, eta
