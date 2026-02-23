import functools
import inspect
import os
from collections.abc import Callable
from typing import Any

from numba import set_num_threads

from aeon.utils.validation import check_n_jobs


def _num_threads_default():
    """Determine the default number of threads to use.

    This code is taken from the Numba source code. The reason for this is that
    in Numba it is defined as an inner function of the `numba.core.threading` module,
    and is not exposed as a public API and therefore it has been copied here.
    """
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


def threaded(func: Callable) -> Callable:
    """
    Temporarily set the global thread count based on an `n_jobs` argument.

    This decorator inspects the decorated function’s signature (and, if needed,
    the calling object’s `n_jobs` attribute) to determine how many threads to
    use. It then calls `set_num_threads(...)` before executing your function,
    and restores the original thread count afterward, ensuring no side effects
    leak out.

    Because it manipulates the global thread pool, it should be applied to
    top-level entry points (e.g. `fit`, `transform`, etc.) rather than to
    individual jitted functions. Any Numba-compiled routines invoked within
    the decorated function will automatically use the thread count set here.

    Parameters
    ----------
    func : Callable
        A function or method that accepts an `n_jobs` parameter (either as a
        keyword or positional arg), or is a bound method on an object with
        a `n_jobs` attribute.

    Returns
    -------
    Callable
        A wrapped version of `func`

    Notes
    -----
    - If `NUMBA_NUM_THREADS` is set in the environment and is a valid integer,
      that value is used as the “original” count for restoration.
    - `n_jobs` can be provided via:
        1. A keyword argument `n_jobs=…`
        2. A positional argument in the signature
        3. An attribute `self.n_jobs` on the first positional arg
    - `check_n_jobs()` is used to translate negative values (e.g. `-1`) or
      other conventions into a concrete thread count.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 1) Capture the “before” thread count
        env_threads = os.environ.get("NUMBA_NUM_THREADS")
        if env_threads is not None and env_threads.isdigit():
            original_thread_count = int(env_threads)
        else:
            original_thread_count = _num_threads_default()

        # 2) Extract the requested n_jobs value
        n_jobs = None

        # A) If passed explicitly as a kwarg, use that
        if "n_jobs" in kwargs:
            n_jobs = kwargs["n_jobs"]
        else:
            # B) Otherwise, inspect the signature to see which positional argument
            #    corresponds to n_jobs, if any.
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if "n_jobs" in params:
                idx = params.index("n_jobs")
                if idx < len(args):
                    # If n_jobs is passed as a positional argument, use that value
                    n_jobs = args[idx]
                else:
                    # Use the function’s declared default if one exists
                    default = sig.parameters["n_jobs"].default
                    if default is not inspect.Parameter.empty:
                        n_jobs = default
            # C) Otherwise, check to see if n_jobs is a method on an object with a
            # .n_jobs attr.
            elif n_jobs is None and args and hasattr(args[0], "n_jobs"):
                n_jobs = args[0].n_jobs

        # 3) Set thread count based on n_jobs
        adjusted = check_n_jobs(n_jobs)
        set_num_threads(adjusted)

        try:
            return func(*args, **kwargs)
        finally:
            # 4) Once finished, restore the original thread count
            set_num_threads(original_thread_count)

    return wrapper
