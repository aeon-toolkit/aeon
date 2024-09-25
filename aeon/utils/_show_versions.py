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
    sys_info = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }

    print("\nSystem:")  # noqa: T001, T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T001, T201

    deps_info = {"aeon": __version__}
    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None

    print("\nPython dependencies:")  # noqa: T001, T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T001, T201
