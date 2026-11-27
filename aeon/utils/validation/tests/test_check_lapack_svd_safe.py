"""Tests for check_lapack_svd_safe."""

import pytest

from aeon.utils.validation import check_lapack_svd_safe

SCIPY_LAPACK_ERRORS = [
    "Indexing a matrix of 3365692186 elements would incur an integer overflow in "
    "LAPACK. Try using numpy.linalg.svd instead.",
    "Indexing a matrix of 2500000000 elements would incur an in integer overflow in "
    "LAPACK. Instead, either use using numpy.linalg.svd or buildSciPy with ILP64 "
    "support.",
]


@pytest.mark.parametrize("message", SCIPY_LAPACK_ERRORS)
def test_check_lapack_svd_safe_raises_informative_error(message):
    """A LAPACK overflow error should be re-raised with an aeon-level message."""
    error = ValueError(message)

    with pytest.raises(ValueError, match="TestEstimator") as exc_info:
        check_lapack_svd_safe(error, 112_186, 30_000, "TestEstimator")

    assert "112186 samples and 30000 features" in str(exc_info.value)
    # The original SciPy message is kept, and the original error is chained.
    assert message in str(exc_info.value)
    assert exc_info.value.__cause__ is error


def test_check_lapack_svd_safe_ignores_other_errors():
    """Errors that are not a LAPACK overflow should be left for the caller."""
    error = ValueError("something else went wrong")

    assert check_lapack_svd_safe(error, 10, 5, "TestEstimator") is None
