"""Utility methods to print system info for debugging.

Adapted from the sklearn show_versions function.
"""

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


def show_versions():
    """Print useful debugging information."""
    print(_show_versions())  # noqa: T001, T201


def _show_versions():
    """Print useful debugging information.

    Returns
    -------
    str
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
        str = f"{str}\n{k:>13}: {stat}"  # noqa: T001, T201
    return str
