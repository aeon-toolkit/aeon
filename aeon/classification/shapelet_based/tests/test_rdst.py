"""RDST tests."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

from aeon.classification.shapelet_based import RDSTClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.utils._lapack import LAPACK_ERROR, _failing_fit


def test_predict_proba():
    """RDST tests for code not covered by standard tests."""
    X = make_example_3d_numpy(return_y=False, n_cases=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    clf = RDSTClassifier(estimator=RandomForestClassifier(n_jobs=1))
    clf.fit(X, y)
    p = clf._predict_proba(X)
    assert p.shape == (10, 2)
    p = clf._predict(X)
    assert p.shape == (10,)


def test_rdst_estimator_attribute_lifecycle():
    """Test that estimator_ is only set after fitting, not before."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=20)
    model = RDSTClassifier(max_shapelets=5)

    assert not hasattr(model, "estimator_")

    model.fit(X, y)

    assert hasattr(model, "estimator_")
    assert not hasattr(model, "_estimator")


def test_rdst_lapack_overflow_gives_informative_error(monkeypatch):
    """The default estimator failing with a LAPACK overflow should be explained."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError(LAPACK_ERROR)
    monkeypatch.setattr(RidgeClassifierCV, "fit", _failing_fit(error))

    with pytest.raises(ValueError, match="RDSTClassifier could not fit") as exc_info:
        RDSTClassifier(max_shapelets=5).fit(X, y)

    assert "10 samples" in str(exc_info.value)
    assert exc_info.value.__cause__ is error


def test_rdst_other_value_errors_are_not_changed(monkeypatch):
    """Errors from the default estimator that are not LAPACK overflows pass through."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError("an unrelated problem")
    monkeypatch.setattr(RidgeClassifierCV, "fit", _failing_fit(error))

    with pytest.raises(ValueError) as exc_info:
        RDSTClassifier(max_shapelets=5).fit(X, y)

    assert exc_info.value is error


def test_rdst_custom_estimator_errors_are_not_changed(monkeypatch):
    """A custom estimator is the user's choice, so its errors are left as they are."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    error = ValueError(LAPACK_ERROR)
    monkeypatch.setattr(RidgeClassifier, "fit", _failing_fit(error))

    with pytest.raises(ValueError) as exc_info:
        RDSTClassifier(max_shapelets=5, estimator=RidgeClassifier()).fit(X, y)

    assert exc_info.value is error
