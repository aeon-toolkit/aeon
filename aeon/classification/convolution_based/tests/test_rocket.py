"""Rocket classifier tests."""

from unittest.mock import patch

from sklearn.linear_model import RidgeClassifier

from aeon.classification.convolution_based import RocketClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_rocket_calls_lapack_check_with_default_estimator():
    """RocketClassifier should call check_lapack_svd_safe when estimator is None."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    clf = RocketClassifier(n_kernels=20)

    with patch(
        "aeon.classification.convolution_based._rocket.check_lapack_svd_safe"
    ) as mock_check:
        clf.fit(X, y)

    mock_check.assert_called_once()
    args, _ = mock_check.call_args
    assert args[2] == "RocketClassifier"


def test_rocket_skips_lapack_check_with_custom_estimator():
    """RocketClassifier should not call check_lapack_svd_safe with custom estimator."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    clf = RocketClassifier(n_kernels=20, estimator=RidgeClassifier())

    with patch(
        "aeon.classification.convolution_based._rocket.check_lapack_svd_safe"
    ) as mock_check:
        clf.fit(X, y)

    mock_check.assert_not_called()
