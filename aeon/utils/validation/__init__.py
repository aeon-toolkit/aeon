"""Validation and checking functions for time series."""

__all__ = [
    "check_n_jobs",
    "check_random_state",
]

import os

import numpy as np
from sklearn.utils import check_random_state as _sklearn_check_random_state


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


def _is_float_or_int_dtype(dtype):
    return isinstance(dtype, np.dtype) and (
        np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
    )


def check_random_state(random_state) -> np.random.RandomState:
    """Check `random_state`, never returning the process-global generator.

    ``sklearn.utils.check_random_state(None)`` returns
    ``np.random.mtrand._rand``, the process-global ``RandomState``. An
    estimator that stores that as fitted state shares a generator with every
    other user of the global stream, so unrelated code drawing random numbers
    silently advances the estimator's own generator.

    Parameters
    ----------
    random_state : int, np.random.RandomState or None
        If int, the seed used by the returned generator. If a ``RandomState``
        other than the global one, it is returned unchanged. If None, a new
        generator seeded from the global stream is returned, so results stay
        non-deterministic without the estimator holding global state.

    Returns
    -------
    np.random.RandomState
        A generator that is not ``np.random.mtrand._rand``.
    """
    if random_state is None or random_state is np.random.mtrand._rand:
        seed = np.random.mtrand._rand.randint(0, np.iinfo(np.int32).max)
        return np.random.RandomState(seed)
    return _sklearn_check_random_state(random_state)
