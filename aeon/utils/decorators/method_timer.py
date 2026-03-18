"""Decorator for timing method calls."""

import functools
import time
from collections.abc import Callable


def method_timer(attr: str) -> Callable:
    """Time a method call.

    A decorator factory that times a method call in milliseconds and writes it to
    an attribute.
    Expects an instance method where the first arg is self.

    Parameters
    ----------
    attr : str
        The name of the attribute to write the timing to.

    Returns
    -------
    Callable
        A decorator that wraps a method.
    """

    def deco(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                raise TypeError(
                    "method_timer expects an instance method (missing self)."
                )
            self = args[0]
            if not hasattr(self, "__dict__") and not hasattr(type(self), "__slots__"):
                raise TypeError(
                    f"method_timer expects first arg to be an instance; can't set "
                    f"{attr!r} on {type(self).__name__}."
                )

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                ms = int((time.perf_counter() - start) * 1000)
                setattr(self, attr, ms)

        return wrapper

    return deco
