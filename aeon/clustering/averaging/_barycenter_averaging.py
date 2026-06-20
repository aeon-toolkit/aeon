"""Compute the barycenter average of time series using an elastic distance."""

__maintainer__ = []

import numpy as np

from aeon.clustering.averaging._ba_petitjean import petitjean_barycenter_average
from aeon.clustering.averaging._ba_subgradient import subgradient_barycenter_average
from aeon.clustering.averaging._ba_utils import VALID_BA_METHODS
from aeon.clustering.averaging._kasba_average import kasba_average


def elastic_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 30,
    tol: float = 1e-5,
    init_barycenter: np.ndarray | str = "mean",
    method: str = "petitjean",
    weights: np.ndarray | None = None,
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    precomputed_medoids_pairwise_distance: np.ndarray | None = None,
    verbose: bool = False,
    random_state: int | None = None,
    decay_rate: float = 0.1,
    previous_cost: float | None = None,
    previous_distance_to_center: np.ndarray | None = None,
    ba_subset_size: float = 0.5,
    return_cost: bool = False,
    return_distances_to_center: bool = False,
    n_jobs: int = 1,
    **kwargs,
):
    """
    Compute the barycenter average of time series using an elastic distance.

    This is a utility function that computes the barycenter average of a collection
    of time series instances using one of several available elastic barycenter
    averaging (EBA) algorithms, specified via the `method` parameter:

    - "petitjean": Adapted version of the original Petitjean DBA algorithm [1]_.
    - "subgradient": Stochastic subgradient DBA algorithm [2]_.
    - "kasba": KASBA algorithm [3]_, a fast stochastic variant that samples subsets
      of time series during each iteration.

    Petitjean is slower but more reliable at converging to the optimal solution.
    Subgradient is faster but not guaranteed to converge optimally. KASBA is
    designed for large datasets, trading off some accuracy for a much faster runtime.

    Parameters
    ----------
    X: np.ndarray of shape (n_cases, n_channels, n_timepoints) or (n_cases,
        n_timepoints)
        Collection of time series instances to average.
    distance : str, default="dtw"
        Distance function to use for averaging. See
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
        - "mean": Uses the mean of the time series.
        - "medoids": Uses the medoid of the time series.
        - "random": Uses a randomly selected time series instance.
    method : {"petitjean", "subgradient", "kasba"}, default="petitjean"
        The algorithm to use for barycenter averaging.
    initial_step_size : float, default=0.05
        Initial step size for gradient-based methods ("subgradient" and "kasba").
    final_step_size : float, default=0.005
        Final step size for the subgradient method (suggested in [2]_).
    decay_rate : float, default=0.1
        Exponential decay rate for the step size in the KASBA method.
    ba_subset_size : float, default=0.5
        Proportion of data sampled in each iteration of the KASBA method.
    weights : np.ndarray of shape (n_cases,), default=None
        Weights for each time series. If None, all series receive weight 1.
    precomputed_medoids_pairwise_distance : np.ndarray of shape (n_cases, n_cases),
        default=None
        Optional precomputed pairwise distance matrix (used when relevant, e.g., for
        "medoids" initialisation). If None, distances are computed on the fly.
    previous_cost : float, default=None
        Used by the KASBA method. Previous total cost, if already known.
    previous_distance_to_center : np.ndarray of shape (n_cases,), default=None
        Used by the KASBA method. Previous distances to center, if already known.
    verbose : bool, default=False
        If True, prints progress information.
    random_state : int or None, default=None
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments passed to the chosen distance function.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        The barycenter (elastic average) of the input time series.

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski.
           "A global averaging method for dynamic time warping, with applications
           to clustering." Pattern Recognition, 44(3), 678–693, 2011.
    .. [2] D. Schultz & B. Jain.
           "Nonsmooth Analysis and Subgradient Methods for Averaging in Dynamic Time
           Warping Spaces." Pattern Recognition, 74, 340–358, 2018.
    .. [3] C. Holder & A. Bagnall.
           "Rock the KASBA: Blazingly Fast and Accurate Time Series Clustering."
           arXiv:2411.17838, 2024.
    """
    if method == "petitjean":
        return petitjean_barycenter_average(
            X,
            distance=distance,
            max_iters=max_iters,
            tol=tol,
            init_barycenter=init_barycenter,
            weights=weights,
            precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
            previous_cost=previous_cost,
            previous_distance_to_center=previous_distance_to_center,
            return_cost=return_cost,
            return_distances_to_center=return_distances_to_center,
            **kwargs,
        )
    elif method == "subgradient":
        return subgradient_barycenter_average(
            X,
            distance=distance,
            max_iters=max_iters,
            tol=tol,
            init_barycenter=init_barycenter,
            initial_step_size=initial_step_size,
            final_step_size=final_step_size,
            weights=weights,
            precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
            previous_cost=previous_cost,
            previous_distance_to_center=previous_distance_to_center,
            return_cost=return_cost,
            return_distances_to_center=return_distances_to_center,
            **kwargs,
        )
    elif method == "kasba":
        return kasba_average(
            X,
            init_barycenter=init_barycenter,
            previous_cost=previous_cost,
            previous_distance_to_center=previous_distance_to_center,
            distance=distance,
            max_iters=max_iters,
            tol=tol,
            weights=weights,
            ba_subset_size=ba_subset_size,
            initial_step_size=initial_step_size,
            decay_rate=decay_rate,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            return_cost=return_cost,
            return_distances_to_center=return_distances_to_center,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Invalid method: {method}. Please use one of the following: "
            f"{VALID_BA_METHODS}"
        )
