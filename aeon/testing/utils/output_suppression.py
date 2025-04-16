"""Utility for suppressing function output."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["suppress_output"]


import sys
from contextlib import contextmanager
from os import devnull


@contextmanager
def suppress_output(suppress_stdout=True, suppress_stderr=True):
    """
    Context manager to suppress stdout and/or stderr output.

    This function redirects standard output (stdout) and standard error (stderr)
    to `devnull`, effectively silencing any print statements or error messages
    within its context.

    Parameters
    ----------
    suppress_stdout : bool, optional, default=True
        If True, redirects stdout to null, suppressing print statements.
    suppress_stderr : bool, optional, default=True
        If True, redirects stderr to null, suppressing error messages.

    Examples
    --------
    Suppressing both stdout and stderr:

    >>> import sys
    >>> with suppress_output():
    ...     print("This will not be displayed")
    ...     print("Error messages will be hidden", file=sys.stderr)

    Suppressing only stdout:

    >>> sys.stderr = sys.stdout # Needed so doctest can capture stderr
    >>> with suppress_output(suppress_stdout=True, suppress_stderr=False):
    ...     print("This will not be shown")
    ...     print("Error messages will still be visible", file=sys.stderr)
    Error messages will still be visible

    Suppressing only stderr:

    >>> with suppress_output(suppress_stdout=False, suppress_stderr=True):
    ...     print("This will be shown")
    ...     print("Error messages will be hidden", file=sys.stderr)
    This will be shown

    Using as a function wrapper:

    Suppressing both stdout and stderr:

    >>> @suppress_output()
    ... def noisy_function():
    ...     print("Noisy output")
    ...     print("Noisy error", file=sys.stderr)
    >>> noisy_function()

    Suppressing only stdout:

    >>> @suppress_output(suppress_stderr=False)
    ... def noisy_function():
    ...     print("Noisy output")
    ...     print("Noisy error", file=sys.stderr)
    >>> noisy_function()
    Noisy error
    """
    with open(devnull, "w") as null:
        stdout = sys.stdout
        stderr = sys.stderr
        try:
            if suppress_stdout:
                sys.stdout = null
            if suppress_stderr:
                sys.stderr = null
            yield
        finally:
            if suppress_stdout:
                sys.stdout = stdout
            if suppress_stderr:
                sys.stderr = stderr
