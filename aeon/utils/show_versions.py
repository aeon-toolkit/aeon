"""Utility methods to print system info for debugging."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["show_versions"]

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from aeon import __version__

deps = [
    "pip",
    "setuptools",
    "scikit-learn",
    "numpy",
    "numba",
    "scipy",
    "pandas",
]


def show_versions(as_str: bool = False) -> str | None:
    """Print useful debugging information.

    Parameters
    ----------
    as_str : bool, default=False
        If True, return the output as a string instead of printing.

    Returns
    -------
    str or None
        The output string if `as_str` is True, otherwise None.

    Notes
    -----
    Adapted from the scikit-learn 1.5.0 show_versions function.
    https://github.com/scikit-learn/scikit-learn/
    Copyright (c) 2007-2024 The scikit-learn developers, BSD-3

    Examples
    --------
    >>> from aeon.utils import show_versions
    >>> vers = show_versions(as_str=True)
    """
    sys_info = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }
    str = "\nSystem:"
    for k, stat in sys_info.items():
        str = f"{str}\n{k:>10}: {stat}"

    deps_info = {"aeon": __version__}
    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    str = f"{str}\nPython dependencies:"
    for k, stat in deps_info.items():
        str = f"{str}\n{k:>13}: {stat}"
    if as_str:
        return str
    print(str)  # noqa: T201
