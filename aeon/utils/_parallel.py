"""Internal helpers for running joblib tasks."""

__maintainer__ = []
__all__ = ["_run_jobs", "_NUMBA_PARALLEL_LOCK"]

import threading

from joblib import Parallel

# numba's default (workqueue) threading layer terminates the process when two
# Python threads enter parallel=True regions concurrently, and
# get_num_threads/set_num_threads pairs race. Estimators that call numba
# parallel functions from joblib threads must hold this lock around the
# set-threads/launch/restore block.
_NUMBA_PARALLEL_LOCK = threading.Lock()


def _run_jobs(tasks, n_jobs, backend=None, prefer=None):
    """Run a list of joblib ``delayed`` tasks, skipping joblib when single-threaded.

    ``tasks`` is an iterable of ``delayed(func)(*args, **kwargs)`` tuples. When
    ``n_jobs == 1`` these are called directly, avoiding joblib's dispatch
    overhead (a ``Parallel`` object plus per-task wrapping) for what is the
    default and dominant case, otherwise they are run in parallel.

    Tasks are consumed in order on both paths, so any random draws made while
    building task arguments happen in the same order sequentially and in
    parallel, keeping results identical between the two.

    Parameters
    ----------
    tasks : iterable of tuple
        Tasks to run, each built with ``joblib.delayed``.
    n_jobs : int
        The number of jobs to run in parallel. Must already be resolved to a
        positive number of processors, e.g. with ``check_n_jobs``.
    backend : str, ParallelBackendBase instance or None, default=None
        The ``backend`` passed to ``joblib.Parallel`` when ``n_jobs != 1``.
    prefer : str or None, default=None
        The ``prefer`` passed to ``joblib.Parallel`` when ``n_jobs != 1``.

    Returns
    -------
    results : list
        The return values of the tasks, in order.
    """
    if n_jobs == 1:
        return [func(*args, **kwargs) for func, args, kwargs in tasks]
    return Parallel(n_jobs=n_jobs, backend=backend, prefer=prefer)(tasks)
