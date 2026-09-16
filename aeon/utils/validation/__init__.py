"""Validation and checking functions for time series."""

__all__ = [
    "check_lapack_svd_safe",
    "check_n_jobs",
]

import os

import numpy as np


def check_n_jobs(n_jobs: int) -> int:
    """Check `n_jobs` parameter according to the scikit-learn convention.

    https://scikit-learn.org/stable/glossary.html#term-n_jobs

    Parameters
    ----------
    n_jobs : int or None
        The number of jobs for parallelization.
        If None or 0, 1 is used.
        If negative, (n_cpus + 1 + n_jobs) is used. In such a case, -1 would use all
        available CPUs and -2 would use all but one. If the number of CPUs used would
        fall under 1, 1 is returned instead.

    Returns
    -------
    n_jobs : int
        The number of threads to be used.
    """
    if n_jobs is None or n_jobs == 0:
        return 1
    elif not isinstance(n_jobs, int):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return max(1, os.cpu_count() + 1 + n_jobs)
    else:
        return n_jobs


def check_lapack_svd_safe(n_samples: int, n_features: int, estimator_name: str) -> None:
    """Raise an informative error if a matrix is too large for LAPACK SVD.

    Matrices with more than ``2**31 - 1`` elements may overflow 32-bit integer
    indexing used internally by LAPACK during SVD-based operations.

    Parameters
    ----------
    n_samples : int
        Number of rows in the matrix.
    n_features : int
        Number of columns in the matrix.
    estimator_name : str
        Name of the calling estimator, used in the error message.

    Raises
    ------
    ValueError
        If ``n_samples * n_features`` exceeds the 32-bit LAPACK indexing limit.
    """
    n_elements = int(n_samples) * int(n_features)
    limit = np.iinfo(np.int32).max

    if n_elements > limit:
        raise ValueError(
            f"{estimator_name} cannot process this data because the "
            f"transformed feature matrix has {n_samples} samples and "
            f"{n_features} features ({n_elements} elements), exceeding "
            f"the {limit} element limit for 32-bit LAPACK indexing during "
            "SVD-based operations. This limitation is independent of "
            "available RAM. Use an estimator or solver that does not rely "
            "on the affected SVD-based operation."
        )
