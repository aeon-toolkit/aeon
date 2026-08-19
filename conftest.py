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
        "--check-soft-dependency-skips",
        action="store_true",
        default=False,
        help=(
            "Fail tests skipped by soft dependency checks. Skips all other tests. "
            "Use only in environments where soft dependencies are installed."
        ),
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

    if config.getoption("--check-soft-dependency-skips"):
        config.pluginmanager.register(
            _CheckSoftDependencySkips(),
            name="check-soft-dependency-skips",
        )


class _CheckSoftDependencySkips:
    _SOFT_DEPENDENCY_TERMS = (
        "soft dependency",
        "soft dependencies",
        "soft-dependency",
        "soft-dependencies",
    )

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_setup(self, item):
        pytest.skip("Tests are not run in current testing setup.")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()

        if self._is_soft_dependency_skip(report):
            skip_reason = self._get_skip_reason(report)
            report.outcome = "failed"
            report.longrepr = (
                "Test skipped because a soft dependency check failed while "
                "--check-soft-dependency-skips is enabled. "
                f"Original skip reason: {skip_reason}"
            )

    @staticmethod
    def _get_skip_reason(report):
        """Extract the skip reason across pytest longrepr formats."""
        if isinstance(report.longrepr, tuple) and len(report.longrepr) >= 3:
            return str(report.longrepr[2])

        return str(report.longrepr)

    @classmethod
    def _is_soft_dependency_skip(cls, report):
        """Check if a soft dependency related test is skipped."""
        if not report.skipped:
            return False

        reason = cls._get_skip_reason(report).lower()
        return any(term in reason for term in cls._SOFT_DEPENDENCY_TERMS)
