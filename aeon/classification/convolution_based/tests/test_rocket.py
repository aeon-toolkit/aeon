"""Rocket classifier tests."""

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.pipeline import Pipeline

from aeon.classification.convolution_based import RocketClassifier
from aeon.testing.data_generation import make_example_3d_numpy

LAPACK_ERROR = (
    "Indexing a matrix of 3365692186 elements would incur an integer overflow in "
    "LAPACK. Try using numpy.linalg.svd instead."
)


def _failing_fit(error):
    """Return a replacement ``fit`` method that raises ``error``."""

    def fit(self, *args, **kwargs):
        raise error

    return fit


def test_rocket_lapack_overflow_gives_informative_error(monkeypatch):
    """The default estimator failing with a LAPACK overflow should be explained."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError(LAPACK_ERROR)
    monkeypatch.setattr(RidgeClassifierCV, "fit", _failing_fit(error))

    with pytest.raises(ValueError, match="RocketClassifier could not fit") as exc_info:
        RocketClassifier(n_kernels=20).fit(X, y)

    assert "10 samples" in str(exc_info.value)
    assert exc_info.value.__cause__ is error


def test_rocket_other_value_errors_are_not_changed(monkeypatch):
    """Errors from the default estimator that are not LAPACK overflows pass through."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError("an unrelated problem")
    monkeypatch.setattr(RidgeClassifierCV, "fit", _failing_fit(error))

    with pytest.raises(ValueError) as exc_info:
        RocketClassifier(n_kernels=20).fit(X, y)

    assert exc_info.value is error


def test_rocket_custom_estimator_errors_are_not_changed(monkeypatch):
    """A custom estimator is the user's choice, so its errors are left as they are."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError(LAPACK_ERROR)
    monkeypatch.setattr(RidgeClassifier, "fit", _failing_fit(error))

    with pytest.raises(ValueError) as exc_info:
        RocketClassifier(n_kernels=20, estimator=RidgeClassifier()).fit(X, y)

    assert exc_info.value is error


def test_rocket_pipeline_attribute_preserved():
    """RocketClassifier should still expose a fitted ``pipeline_`` after fit."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=12, random_state=0
    )
    clf = RocketClassifier(n_kernels=20, random_state=0)
    clf.fit(X, y)

    assert isinstance(clf.pipeline_, Pipeline)
    assert clf.pipeline_.steps[-1][1] is clf.estimator_
    assert np.array_equal(clf.pipeline_.predict(X), clf.predict(X))
