"""Main configuration file for pytest.

Contents:
Adds a --prtesting option to pytest.
This allows for smaller parameter matrices to be used for certain tests and enables
sub-sampling in the tests (for shorter runtime) ensuring that each estimators full
tests are run on each operating system at least once, and on each python version at
least once, but not necessarily on each operating system / python version combination.
"""

__maintainer__ = []


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
    import os

    # Must be called before any numpy imports
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    import numba

    from aeon.testing import test_config
    from aeon.utils.validation._dependencies import _check_soft_dependencies

    numba.set_num_threads(1)

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
        test_config.PR_TESTING = True
