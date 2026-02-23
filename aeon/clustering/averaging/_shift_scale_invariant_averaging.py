"""Shift-invariant average."""

import numpy as np
from joblib import Parallel, delayed

from aeon.distances import shift_scale_invariant_best_shift
from aeon.utils.validation import check_n_jobs


def shift_invariant_average(
    X: np.ndarray,
    initial_center: np.ndarray | None = None,
    max_shift: int | None = None,
    n_jobs: int = 1,
):
    """
    Shift-invariant average with parallel processing support.

    Computes a barycenter that is invariant to circular/time shifts by aligning each
    instance in `X` to a common center using a shift-invariant distance. Using these
    optimal shifts, a covariance matrix :math:`M` is constructed. Eigen decomposition
    of :math:`M` is then performed, and the eigenvector corresponding to the smallest
    eigenvalue is used as the new centroid.

    Parameters
    ----------
    X: np.ndarray of shape (n_instances, n_dims, n_timepoints)
        Collection of time series to average.
    initial_center : np.ndarray of shape (n_dims, n_timepoints), default=None
        Initial center used for alignment. If None, the arithmetic mean of `X`
        over the first axis is used.
    max_shift : int or None, default=None
        Maximum shift allowed in the alignment path. If None, then `max_shift` is set
        to `min(x.shape[1], y.shape[1])`.
    n_jobs : int, default=1
        The number of parallel jobs to run. Use -1 for all available processors.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`aeon.distances.shift_scale_invariant_best_shift`.

    Returns
    -------
    np.ndarray of shape (n_dims, n_timepoints)
        The shift-invariant barycenter.
    """
    n_jobs = check_n_jobs(n_jobs)
    if initial_center is None:
        initial_center = X.mean(axis=0)

    if max_shift is None:
        max_shift = np.min(X.shape[-1])
    optimal_shifts = np.zeros_like(X)

    n_instances, n_dims, n_timepoints = X.shape

    def compute_shift(x_instance):
        if not initial_center.any():
            return x_instance
        else:
            _, curr_shift = shift_scale_invariant_best_shift(
                initial_center, x_instance, max_shift
            )
            return curr_shift

    temp_shifts = Parallel(n_jobs=n_jobs)(
        delayed(compute_shift)(X[i]) for i in range(n_instances)
    )
    for i in range(n_instances):
        optimal_shifts[i] = temp_shifts[i]

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
