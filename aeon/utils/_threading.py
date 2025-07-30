import functools
import inspect
import os
from typing import Any, Callable

from numba import set_num_threads

from aeon.utils.validation import check_n_jobs


def threaded(func: Callable) -> Callable:
    """Set thread count based on n_jobs parameter and restore it afterward.

    A decorator that sets the number of threads based on the n_jobs parameter
    passed to the function, and restores the original thread count afterward.

    The decorated function is expected to have a 'n_jobs' parameter.
    """

    def num_threads_default():
        try:
            sched_getaffinity = os.sched_getaffinity
        except AttributeError:
            pass
        else:
            return max(1, len(sched_getaffinity(0)))

        cpu_count = os.cpu_count()
        if cpu_count is not None:
            return max(1, cpu_count)

        return 1

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        NUMBA_ENV_THREADS = os.environ.get("NUMBA_NUM_THREADS")

        if NUMBA_ENV_THREADS is not None and NUMBA_ENV_THREADS.isdigit():
            original_thread_count = int(NUMBA_ENV_THREADS)
        else:
            original_thread_count = num_threads_default()

        n_jobs = None
        if "n_jobs" in kwargs:
            n_jobs = kwargs["n_jobs"]
        else:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            if "n_jobs" in param_names:
                n_jobs_index = param_names.index("n_jobs")
                if n_jobs_index < len(args):
                    n_jobs = args[n_jobs_index]
                else:
                    default = sig.parameters["n_jobs"].default
                    n_jobs = default if default is not inspect.Parameter.empty else None

            if n_jobs is None and args and hasattr(args[0], "n_jobs"):
                # This gets n_jobs if it belongs to a object (i.e. self.n_jobs)
                n_jobs = args[0].n_jobs

        adjusted_n_jobs = check_n_jobs(n_jobs)
        set_num_threads(adjusted_n_jobs)

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            try:
                set_num_threads(original_thread_count)
            except Exception:
                raise ValueError(
                    f"Failed to restore original thread count: {original_thread_count} "
                    f"type {type(original_thread_count)} \n\n "
                    f"\n\n Kwargs: {kwargs}"
                )

    return wrapper
