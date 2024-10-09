"""Utility for suppressing function output."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["suppress_output"]


import sys
from contextlib import contextmanager
from os import devnull


@contextmanager
def suppress_output(suppress_stdout=True, suppress_stderr=True):
    """Redirects stdout and/or stderr to devnull."""
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
