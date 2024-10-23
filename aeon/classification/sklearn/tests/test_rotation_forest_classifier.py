"""Rotation Forest test code."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from aeon.classification.sklearn import RotationForestClassifier
from aeon.datasets import load_unit_test
from aeon.testing.data_generation import make_example_3d_numpy


def test_rotf_output():
    """Test of RotF contracting and train estimate on test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, y_test = load_unit_test(split="test", return_type="numpy2d")

    rotf = RotationForestClassifier(
        n_estimators=10,
        pca_solver="randomized",
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    expected = [
        [0.8, 0.2],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.8, 0.2],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.9, 0.1],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.8, 0.2],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.4, 0.6],
    ]

    np.testing.assert_array_almost_equal(
        expected, rotf.predict_proba(X_test[:15]), decimal=4
    )


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


def test_rotf_input():
    """Test RotF with incorrect input."""
    rotf = RotationForestClassifier()
    X2 = rotf._check_X(np.random.random((10, 1, 100)))
    assert X2.shape == (10, 100)
    with pytest.raises(
        ValueError, match="RotationForestClassifier is not a time series classifier"
    ):
        rotf._check_X(np.random.random((10, 10, 100)))
    with pytest.raises(
        ValueError, match="RotationForestClassifier is not a time series classifier"
    ):
        rotf._check_X([[1, 2, 3], [4, 5], [6, 7, 8]])
    X, y = make_example_3d_numpy()
