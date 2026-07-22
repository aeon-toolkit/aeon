"""Rotation Forest test code."""

import numpy as np
from sklearn.metrics import accuracy_score

from aeon.classification.sklearn import RotationForestClassifier
from aeon.datasets import load_unit_test


def test_rotf_output():
    """Test RotF probability estimates match expected values on unit test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, _ = load_unit_test(split="test", return_type="numpy2d")

    rotf = RotationForestClassifier(
        n_estimators=10,
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    expected = np.array(
        [
            [0.9, 0.1],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.9, 0.1],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.9, 0.1],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.9, 0.1],
            [1.0, 0.0],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.6, 0.4],
        ]
    )

    np.testing.assert_array_almost_equal(
        rotf.predict_proba(X_test[: len(expected)]), expected, decimal=4
    )


def test_contracted_rotf():
    """Test contracted RotF stays within the contract and keeps its accuracy."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, y_test = load_unit_test(split="test", return_type="numpy2d")

    contract_max_n_estimators = 5

    rotf = RotationForestClassifier(
        time_limit_in_minutes=5,
        contract_max_n_estimators=contract_max_n_estimators,
        random_state=0,
    )
    rotf.fit(X_train, y_train)
    assert 0 < len(rotf.estimators_) <= contract_max_n_estimators

    y_pred = rotf.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_test.shape

    acc = accuracy_score(y_test, y_pred)
    np.testing.assert_almost_equal(acc, 0.909, decimal=4)


def test_rotf_fit_predict():
    """Test RotF fit_predict_proba returns train probability estimates."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    n_classes = len(np.unique(y_train))
    n_estimators = 5

    rotf = RotationForestClassifier(
        n_estimators=n_estimators,
        random_state=0,
    )

    y_proba = rotf.fit_predict_proba(X_train, y_train)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (len(y_train), n_classes)
    assert len(rotf.estimators_) == n_estimators
    assert rotf._is_fitted

    y_proba = rotf.predict_proba(X_train)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (len(y_train), n_classes)
