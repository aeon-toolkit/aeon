"""Main configuration file for pytest.

Contents:
Adds a --prtesting option to pytest.
This allows for smaller parameter matrices to be used for certain tests and enables
sub-sampling in the tests (for shorter runtime) ensuring that each estimators full
tests are run on each operating system at least once, and on each python version at
least once, but not necessarily on each operating system / python version combination.
"""

__maintainer__ = ["MatthewMiddlehurst"]

import pytest


def _is_soft_dependency_skip(report):
    if not report.skipped:
        return False

    skip_reason = (
        report.longrepr[2]
        if isinstance(report.longrepr, tuple)
        else str(report.longrepr)
    )
    return "soft dependency" in skip_reason.lower()


def pytest_addoption(parser):
    """Pytest command line parser options adder."""
    parser.addoption(
        "--nonumba",
        default=False,
        help=("Disable numba via the NUMBA_DISABLE_JIT environment variable."),
    )
    parser.addoption(
        "--enablethreading",
        default=False,
        help=(
            "Allow threading and skip setting number of threads to 1 for various "
            "libraries and environment variables."
        ),
    )
    parser.addoption(
        "--prtesting",
        default=False,
        help=(
            "Toggle for PR test configuration. Uses smaller parameter matrices for "
            "test generation and sub-samples estimators in tests by workflow os/py "
            "version."
        ),
    )
    parser.addoption(
        "--fail-soft-dependency-skips",
        action="store_true",
        default=False,
        help=(
            "Fail tests skipped by soft dependency checks. Use only in environments "
            "where soft dependencies are installed."
        ),
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Turn soft dependency skips into failures when requested."""
    outcome = yield
    report = outcome.get_result()

    if item.config.getoption(
        "--fail-soft-dependency-skips"
    ) and _is_soft_dependency_skip(report):
        report.outcome = "failed"
        report.longrepr = (
            "Test skipped because a soft dependency check failed while "
            "--fail-soft-dependency-skips is enabled. "
            f"Original skip reason: {report.longrepr}"
        )


def pytest_configure(config):
    """Pytest configuration preamble."""
    import os

    if config.getoption("--nonumba") in [True, "True", "true"]:
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    if config.getoption("--enablethreading") in [True, "True", "true"]:
        from aeon.testing import testing_config

        testing_config.MULTITHREAD_TESTING = True
    else:
        # Must be called before any numpy imports
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

        import numba

        numba.set_num_threads(1)

        from aeon.utils.validation._dependencies import _check_soft_dependencies

        if _check_soft_dependencies("tensorflow", severity="none"):
            from tensorflow.config.threading import (
                set_inter_op_parallelism_threads,
                set_intra_op_parallelism_threads,
            )

            set_inter_op_parallelism_threads(1)
            set_intra_op_parallelism_threads(1)

        if _check_soft_dependencies("torch", severity="none"):
            import torch

            torch.set_num_threads(1)

    if config.getoption("--prtesting") in [True, "True", "true"]:
        from aeon.testing import testing_config

        testing_config.PR_TESTING = True
