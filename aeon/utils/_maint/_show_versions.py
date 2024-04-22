"""Utility methods to print system info for debugging.

adapted from :func:`sklearn.show_versions`
"""

__maintainer__ = []
__all__ = ["show_versions"]

import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def _get_sys_info():
    """
    System information.

    Return
    ------
    sys_info : dict
        system and Python version information
    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    """
    Overview of the installed version of main dependencies.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = [
        "pip",
        "setuptools",
        "scikit-learn",
        "aeon",
        "statsmodels",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "joblib",
        "numba",
        "pmdarima",
        "tsfresh",
    ]

    def get_version(module):
        return module.__version__

    deps_info = {}

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None

    return deps_info


def show_versions():
    """Print useful debugging information."""
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")  # noqa: T001, T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T001, T201

    print("\nPython dependencies:")  # noqa: T001, T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T001, T201
