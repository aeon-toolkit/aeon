"""MultiRocketHydra classifier tests."""

from unittest.mock import patch

from sklearn.linear_model import RidgeClassifier

from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_mrhydra_calls_lapack_check_with_default_estimator():
    """Check LAPACK safety when using the default estimator.

    MultiRocketHydra should call check_lapack_svd_safe when estimator is None.
    """
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    clf = MultiRocketHydraClassifier(n_kernels=2, n_groups=2)

    with patch(
        "aeon.classification.convolution_based._mr_hydra.check_lapack_svd_safe"
    ) as mock_check:
        clf.fit(X, y)

    mock_check.assert_called_once()
    args, _ = mock_check.call_args
    assert args[2] == "MultiRocketHydraClassifier"


def test_mrhydra_skips_lapack_check_with_custom_estimator():
    """Skip the LAPACK safety check with a custom estimator.

    MultiRocketHydra should not call check_lapack_svd_safe when a custom
    estimator is supplied.
    """
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    clf = MultiRocketHydraClassifier(n_kernels=2, estimator=RidgeClassifier())

    with patch(
        "aeon.classification.convolution_based._mr_hydra.check_lapack_svd_safe"
    ) as mock_check:
        clf.fit(X, y)

    mock_check.assert_not_called()
