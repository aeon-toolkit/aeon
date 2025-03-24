"""Tests for the aeon package and testing module utilties."""

import pkgutil

import aeon

# collect all modules
ALL_AEON_MODULES = pkgutil.walk_packages(aeon.__path__, aeon.__name__ + ".")
ALL_AEON_MODULES = [x[1] for x in ALL_AEON_MODULES]

ALL_AEON_MODULES_NO_TESTS = [
    x for x in ALL_AEON_MODULES if not any(part == "tests" for part in x.split("."))
]
