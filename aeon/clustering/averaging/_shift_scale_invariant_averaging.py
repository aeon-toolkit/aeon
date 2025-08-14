"""Shift-invariant average."""

import numpy as np
from joblib import Parallel, delayed

from aeon.distances import shift_scale_invariant_best_shift
from aeon.utils.validation import check_n_jobs


def _compute_shift_for_instance(args):
    X_i, initial_center, max_shift = args
    if not initial_center.any():
        return X_i
    else:
        _, curr_shift = shift_scale_invariant_best_shift(initial_center, X_i, max_shift)
        return curr_shift


def shift_invariant_average(
    X: np.ndarray,
    initial_center: np.ndarray | None = None,
    max_shift: int | None = None,
    n_jobs: int = 1,
    **kwargs,
):
    """Compute the shift-invariant average of time series.

    Parameters
    ----------
    X : np.ndarray
        Time series instances to compute average from,
        shape (n_cases, n_channels, n_timepoints) or (n_cases, n_timepoints).
    initial_center : np.ndarray or None, default=None
        Initial center to use for the shift-invariant average. If None, uses the mean
        of X.
    max_shift : int or None, default=None
        Maximum shift allowed in the alignment path. If None, uses min(X.shape[-1]).
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. If 1, then the function is executed in a single
        thread. If greater than 1, then the function is executed in parallel.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        The shift-invariant average of the time series.
    """
    n_jobs = check_n_jobs(n_jobs)

    if initial_center is None:
        initial_center = X.mean(axis=0)

    if max_shift is None:
        max_shift = np.min(X.shape[-1])
    n_instances, n_dims, n_timepoints = X.shape
    optimal_shifts = np.zeros_like(X)

    args_list = [(X[i], initial_center, max_shift) for i in range(X.shape[0])]

    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_shift_for_instance)(args) for args in args_list
        )
        for i, result in enumerate(results):
            optimal_shifts[i] = result
    else:
        for i in range(X.shape[0]):
            optimal_shifts[i] = _compute_shift_for_instance(args_list[i])

    if optimal_shifts.shape[0] == 0:
        return np.zeros((n_dims, n_timepoints))

    normalised_shifts = optimal_shifts / np.sqrt(
        np.sum(optimal_shifts**2, axis=2)
    ).reshape(n_instances, n_dims, 1)

    M = np.zeros((n_dims, n_timepoints, n_timepoints))
    new_center = np.zeros((n_dims, n_timepoints))
    for d in range(n_dims):
        M[d] = np.dot(
            normalised_shifts[:, d, :].T, normalised_shifts[:, d, :]
        ) - n_instances * np.eye(n_timepoints)

        eigvals, eigvecs = np.linalg.eig(M[d])
        new_center[d] = np.real(eigvecs[:, np.argmax(eigvals)])

        if np.sum(new_center[d]) < 0:
            new_center[d] = -new_center[d]

    return new_center
