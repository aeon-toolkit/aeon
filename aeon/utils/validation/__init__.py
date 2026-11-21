"""Validation and checking functions for time series."""

__all__ = [
    "check_lapack_svd_safe",
    "check_n_jobs",
]

import os

import numpy as np

_LAPACK_OVERFLOW_TEXT = "integer overflow in LAPACK"


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

    Examples
    --------
    >>> from aeon.utils.validation import check_n_jobs
    >>> check_n_jobs(None)
    1
    >>> check_n_jobs(0)
    1
    >>> check_n_jobs(2)
    2
    """
    if n_jobs is None or n_jobs == 0:
        return 1
    elif not isinstance(n_jobs, int):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return max(1, os.cpu_count() + 1 + n_jobs)
    else:
        return n_jobs


def check_lapack_svd_safe(
    error: ValueError, n_samples: int, n_features: int, estimator_name: str
) -> None:
    """Raise an informative error if ``error`` is a LAPACK integer overflow.

    Parameters
    ----------
    error : ValueError
        The error caught while fitting.
    n_samples : int
        Number of rows in the matrix passed to the estimator.
    n_features : int
        Number of columns in the matrix passed to the estimator.
    estimator_name : str
        Name of the calling estimator, used in the error message.

    Raises
    ------
    ValueError
        If ``error`` reports a LAPACK integer overflow. The original error
        is chained as the cause.

    Examples
    --------
    >>> from aeon.utils.validation import check_lapack_svd_safe
    >>> try:
    ...     raise ValueError("an unrelated error")
    ... except ValueError as error:
    ...     check_lapack_svd_safe(error, 100, 10, "MyEstimator")  # does nothing
    """
    if _LAPACK_OVERFLOW_TEXT not in str(error):
        return

    raise ValueError(
        f"{estimator_name} could not fit its default estimator on the transformed "
        f"feature matrix ({n_samples} samples and {n_features} features) due to a "
        "LAPACK integer overflow. Use an estimator that does not rely on SVD, "
        "such as sklearn.linear_model.RidgeClassifier(solver='lsqr'). "
        f"Original error: {error}"
    ) from error


def _is_float_or_int_dtype(dtype):
    return isinstance(dtype, np.dtype) and (
        np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
    )
