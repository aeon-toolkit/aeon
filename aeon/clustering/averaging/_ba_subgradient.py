__maintainer__ = []

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
    init_barycenter: np.ndarray | str = "mean",
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    weights: np.ndarray | None = None,
    precomputed_medoids_pairwise_distance: np.ndarray | None = None,
    verbose: bool = False,
    random_state: int | None = None,
    **kwargs,
):
    """
    Compute the stochastic subgradient barycenter average of time series.

    This implements a stochastic subgradient variant of DBA (cf. [2]_), which
    updates the barycenter using subgradients computed per time series. It is
    typically faster than the original Petitjean DBA but is not guaranteed to find
    the optimal solution.

    Parameters
    ----------
    X: np.ndarray of shape (n_cases, n_channels, n_timepoints) or (n_cases,
        n_timepoints)
        Collection of time series to average. If 2D, it is internally reshaped to
        (n_cases, 1, n_timepoints).
    distance : str, default="dtw"
        Distance function used during averaging. See
        :func:`aeon.distances.get_distance_function` for valid options.
    max_iters : int, default=30
        Maximum number of update iterations.
    tol : float, default=1e-5
        Early-stopping tolerance: if the decrease in cost between iterations is
        smaller than this value, the procedure stops.
    init_barycenter: {"mean", "medoids", "random"} or np.ndarray of shape (n_channels,
        n_timepoints), default="mean"
        Initial barycenter. If a string is provided, it specifies the initialisation
        strategy. If an array is provided, it is used directly as the starting
        barycenter.
    initial_step_size : float, default=0.05
        Initial step size for the subgradient descent updates (suggested in [2]_).
    final_step_size : float, default=0.005
        Final step size for the subgradient descent updates (suggested in [2]_).
    weights : np.ndarray of shape (n_cases,), default=None
        Weights for each time series. If None, all series receive weight 1.
    precomputed_medoids_pairwise_distance : np.ndarray of shape (n_cases, n_cases),
        default=None
        Optional precomputed pairwise distance matrix (used when relevant, e.g., for
        "medoids" initialisation). If None, distances are computed on the fly.
    verbose : bool, default=False
        If True, prints progress information.
    random_state : int or None, default=None
        Random seed used where applicable (e.g., for shuffling/initialisation).
    **kwargs
        Additional keyword arguments forwarded to the chosen distance function.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        The barycenter (stochastic subgradient DBA average) of the input time series.

    References
    ----------
    . [1] F. Petitjean, A. Ketterlin & P. Gancarski. "A global averaging method
           for dynamic time warping, with applications to clustering."
           Pattern Recognition, 44(3), 678–693, 2011.
    . [2] D. Schultz & B. Jain. "Nonsmooth Analysis and Subgradient Methods
           for Averaging in Dynamic Time Warping Spaces."
           Pattern Recognition, 74, 340–358, 2018.
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

    random_state = check_random_state(random_state)

    cost_prev = np.inf
    if distance == "wdtw" or distance == "wddtw":
        if "g" not in kwargs:
            kwargs["g"] = 0.05

    current_step_size = initial_step_size
    X_size = _X.shape[0]
    for i in range(max_iters):
        shuffled_indices = random_state.permutation(X_size)
        barycenter, cost, current_step_size = _ba_one_iter_subgradient(
            barycenter,
            _X,
            shuffled_indices,
            distance,
            initial_step_size,
            final_step_size,
            current_step_size,
            weights,
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
    current_step_size: float = 0.05,
    weights: np.ndarray | None = None,
    iteration: int = 0,
    window: float | None = None,
    g: float = 0.0,
    epsilon: float | None = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 30,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: np.ndarray | None = None,
    transformed_y: np.ndarray | None = None,
    gamma: float = 1.0,
):

    X_size, X_dims, X_timepoints = X.shape
    cost = 0.0
    # Only update current_step_size on the first iteration
    step_size_reduction = 0.0
    if iteration == 0:
        step_size_reduction = (initial_step_size - final_step_size) / X_size

    barycenter_copy = np.copy(barycenter)

    for i in shuffled_indices:
        curr_ts = X[i]
        curr_alignment, curr_cost = _get_alignment_path(
            center=barycenter,
            ts=curr_ts,
            distance=distance,
            window=window,
            g=g,
            epsilon=epsilon,
            nu=nu,
            lmbda=lmbda,
            independent=independent,
            c=c,
            descriptor=descriptor,
            reach=reach,
            warp_penalty=warp_penalty,
            transformation_precomputed=transformation_precomputed,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
            gamma=gamma,
        )

        new_ba = np.zeros((X_dims, X_timepoints))
        for j, k in curr_alignment:
            new_ba[:, k] += barycenter_copy[:, k] - curr_ts[:, j]

        barycenter_copy -= (2.0 * current_step_size) * new_ba * weights[i]

        current_step_size -= step_size_reduction
        cost = curr_cost * weights[i]
    return barycenter_copy, cost, current_step_size
