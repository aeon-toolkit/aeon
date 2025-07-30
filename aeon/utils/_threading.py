import functools
import inspect
import os
import threading
from typing import Any, Callable

from numba import set_num_threads

from aeon.utils.validation import check_n_jobs


def threaded(func: Callable) -> Callable:
    """Set thread count based on n_jobs parameter and restore it afterward.

    A decorator that sets the number of threads based on the n_jobs parameter
    passed to the function, and restores the original thread count afterward.

    The decorated function is expected to have a 'n_jobs' parameter.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        numba_env_threads = os.environ.get("NUMBA_NUM_THREADS")

        if numba_env_threads is not None and numba_env_threads.isdigit():
            original_thread_count = int(numba_env_threads)
        else:
            active_count = threading.active_count()
            if isinstance(active_count, int):
                original_thread_count = threading.active_count()
            else:
                original_thread_count = 1

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
            if original_thread_count is None:
                original_thread_count = 1
            if not isinstance(original_thread_count, int):
                raise TypeError(
                    f"Expected original_thread_count to be an int, got type: "
                    f"{type(original_thread_count)} value: {original_thread_count}"
                )
            set_num_threads(original_thread_count)

    return wrapper
