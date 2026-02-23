"""Validation and checking functions for time series."""

__all__ = [
    "check_n_jobs",
]

import os


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
