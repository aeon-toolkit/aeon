"""MultiRocketHydra classifier tests."""

import pytest
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.utils._lapack import LAPACK_ERROR, _failing_fit
from aeon.utils.validation._dependencies import _check_soft_dependencies

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency not available",
)


def test_mrhydra_lapack_overflow_gives_informative_error(monkeypatch):
    """The default estimator failing with a LAPACK overflow should be explained."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError(LAPACK_ERROR)
    monkeypatch.setattr(RidgeClassifierCV, "fit", _failing_fit(error))

    with pytest.raises(
        ValueError, match="MultiRocketHydraClassifier could not fit"
    ) as exc_info:
        MultiRocketHydraClassifier(n_kernels=2, n_groups=2).fit(X, y)

    assert "10 samples" in str(exc_info.value)
    assert exc_info.value.__cause__ is error


def test_mrhydra_other_value_errors_are_not_changed(monkeypatch):
    """Errors from the default estimator that are not LAPACK overflows pass through."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError("an unrelated problem")
    monkeypatch.setattr(RidgeClassifierCV, "fit", _failing_fit(error))

    with pytest.raises(ValueError) as exc_info:
        MultiRocketHydraClassifier(n_kernels=2, n_groups=2).fit(X, y)

    assert exc_info.value is error


def test_mrhydra_custom_estimator_errors_are_not_changed(monkeypatch):
    """A custom estimator is the user's choice, so its errors are left as they are."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError(LAPACK_ERROR)
    monkeypatch.setattr(RidgeClassifier, "fit", _failing_fit(error))

    with pytest.raises(ValueError) as exc_info:
        MultiRocketHydraClassifier(
            n_kernels=2, n_groups=2, estimator=RidgeClassifier()
        ).fit(X, y)

    assert exc_info.value is error
