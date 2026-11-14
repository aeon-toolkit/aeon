"""Internal helpers for running joblib tasks and numba parallel code."""

__maintainer__ = []
__all__ = [
    "_run_jobs",
    "_numba_threads",
    "_NUMBA_PARALLEL_LOCK",
    "_NUMBA_RANDOM_LOCK",
]

import threading
from contextlib import contextmanager

from joblib import Parallel
from numba import config, get_num_threads, set_num_threads

# numba's workqueue threading layer terminates the process when two Python
# threads enter parallel=True regions concurrently. Estimators that call numba
# parallel functions from joblib threads can hold this lock around the launch
# to prevent that, at the cost of running those launches one at a time.
_NUMBA_PARALLEL_LOCK = threading.Lock()

# np.random in compiled numba code uses a thread-local generator, but with
# NUMBA_DISABLE_JIT=1 the same functions run on numpy's global RandomState,
# which all Python threads share. Estimators that call a numba function which
# seeds np.random and then draws from it, from joblib threads, must hold this
# lock around the call so concurrent draws do not interleave.
_NUMBA_RANDOM_LOCK = threading.Lock()


@contextmanager
def _numba_threads(n_jobs):
    """Run numba parallel code on ``n_jobs`` threads, restoring the count after.

    numba's thread count is per Python thread, so this is safe to use from
    joblib threads. ``n_jobs`` is capped at numba's thread pool size, above
    which ``set_num_threads`` raises. It does not stop parallel launches from
    different threads overlapping, see ``_NUMBA_PARALLEL_LOCK``.

    Parameters
    ----------
    n_jobs : int
        The number of threads to use. Must already be resolved to a positive
        number, e.g. with ``check_n_jobs``.
    """
    prev_threads = get_num_threads()
    try:
        set_num_threads(min(n_jobs, config.NUMBA_NUM_THREADS))
        yield
    finally:
        set_num_threads(prev_threads)


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
