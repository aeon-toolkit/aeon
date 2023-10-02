# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Main configuration file for pytest.

Contents:
Adds a --prtesting option to pytest.
This allows for smaller parameter matrices to be used for certain tests and enables
sub-sampling in the tests (for shorter runtime) ensuring that each estimators full
tests are run on each operating system at least once, and on each python version at
least once, but not necessarily on each operating system / python version combination.
"""

__author__ = ["fkiraly", "MatthewMiddlehurst"]

from aeon.tests import _config


def pytest_addoption(parser):
    """Pytest command line parser options adder."""
    parser.addoption(
        "--prtesting",
        default=False,
        help=(
            "Toggle for PR test configuration. Uses smaller parameter matrices for "
            "test generation and sub-samples estimators in tests by workflow os/py "
            "version."
        ),
    )


def pytest_configure(config):
    """Pytest configuration preamble."""
    if config.getoption("--prtesting") in [True, "True", "true"]:
        _config.PR_TESTING = True
