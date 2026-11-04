"""Tests for check_lapack_svd_safe."""

import numpy as np
import pytest

from aeon.utils.validation import check_lapack_svd_safe


def test_check_lapack_svd_safe_under_limit():
    """Matrices under the 32-bit LAPACK element limit should not raise."""
    limit = np.iinfo(np.int32).max
    n_samples = 1000
    n_features = limit // n_samples - 1

    check_lapack_svd_safe(n_samples, n_features, "TestEstimator")


def test_check_lapack_svd_safe_at_limit():
    """Exactly at the limit should not raise."""
    limit = np.iinfo(np.int32).max
    n_samples = 1
    n_features = limit

    check_lapack_svd_safe(n_samples, n_features, "TestEstimator")


def test_check_lapack_svd_safe_over_limit():
    """Matrices exceeding the limit should raise an informative ValueError."""
    n_samples = 112_186
    n_features = 30_000

    with pytest.raises(ValueError, match="TestEstimator"):
        check_lapack_svd_safe(n_samples, n_features, "TestEstimator")


def test_check_lapack_svd_safe_message_mentions_element_count():
    """The raised error should surface sample/feature/element counts."""
    n_samples = 200_000
    n_features = 20_000

    with pytest.raises(ValueError, match=str(n_samples * n_features)):
        check_lapack_svd_safe(n_samples, n_features, "TestEstimator")
