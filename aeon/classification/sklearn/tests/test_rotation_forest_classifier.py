"""Rotation Forest test code."""

import numpy as np
from sklearn.metrics import accuracy_score

from aeon.classification.sklearn import RotationForestClassifier
from aeon.datasets import load_unit_test


def test_contracted_rotf():
    """Test of RotF contracting and train estimate on test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, y_test = load_unit_test(split="test", return_type="numpy2d")

    rotf = RotationForestClassifier(
        time_limit_in_minutes=5,
        contract_max_n_estimators=5,
        random_state=0,
    )
    rotf.fit(X_train, y_train)
    assert len(rotf.estimators_) > 0

    y_pred = rotf.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(y_test)

    acc = accuracy_score(y_test, y_pred)
    np.testing.assert_almost_equal(acc, 0.909, decimal=4)


def test_rotf_fit_predict():
    """Test of RotF fit_predict on test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")

    rotf = RotationForestClassifier(
        n_estimators=5,
        random_state=0,
    )

    y_proba = rotf.fit_predict_proba(X_train, y_train)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (len(y_train), 2)
    assert len(rotf.estimators_) > 0
    assert rotf._is_fitted

    y_proba = rotf.predict_proba(X_train)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (len(y_train), 2)
