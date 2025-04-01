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

    >>> with suppress_output():
    ...     print("This will not be displayed")
    ...     import warnings
    ...     warnings.warn("This warning will be hidden", UserWarning)

    Suppressing only stdout:

    >>> with suppress_output(suppress_stdout=True, suppress_stderr=False):
    ...     print("This will not be shown")
    ...     import sys
    ...     sys.stderr.write("Error messages will still be visible")

    Suppressing only stderr:

    >>> with suppress_output(suppress_stdout=False, suppress_stderr=True):
    ...     print("This will be shown")
    ...     import sys
    ...     sys.stderr.write("This error message will be hidden")
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
