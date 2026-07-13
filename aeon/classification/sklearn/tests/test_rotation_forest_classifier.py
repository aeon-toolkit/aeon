"""Rotation Forest test code."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from aeon.classification.sklearn import RotationForestClassifier
from aeon.datasets import load_unit_test


def test_rotf_output():
    """Test RotF probability estimates match expected values on unit test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, y_test = load_unit_test(split="test", return_type="numpy2d")

    rotf = RotationForestClassifier(
        n_estimators=10,
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    # expected values changed when the group PCA moved to an exact
    # eigendecomposition
    expected = [
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

    np.testing.assert_array_almost_equal(
        expected, rotf.predict_proba(X_test[:15]), decimal=4
    )


def test_rotf_pca_solver_is_noop():
    """Test pca_solver is retained for compatibility but has no effect."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, _ = load_unit_test(split="test", return_type="numpy2d")

    rotf_full = RotationForestClassifier(
        n_estimators=10,
        pca_solver="full",
        random_state=0,
    )
    rotf_randomized = RotationForestClassifier(
        n_estimators=10,
        pca_solver="randomized",
        random_state=0,
    )

    rotf_full.fit(X_train, y_train)
    rotf_randomized.fit(X_train, y_train)

    np.testing.assert_array_equal(
        rotf_full.predict_proba(X_test[:15]),
        rotf_randomized.predict_proba(X_test[:15]),
    )


def test_contracted_rotf():
    """Test RotF time contract produces a usable ensemble on unit test data."""
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
    """Test RotF fit_predict_proba returns train probability estimates."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    n_classes = len(np.unique(y_train))

    rotf = RotationForestClassifier(
        n_estimators=5,
        random_state=0,
    )

    y_proba = rotf.fit_predict_proba(X_train, y_train)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (len(y_train), n_classes)
    assert len(rotf.estimators_) > 0
    assert rotf._is_fitted

    y_proba = rotf.predict_proba(X_train)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (len(y_train), n_classes)


def test_rotf_input():
    """Test RotF rejects unsupported input shapes and degenerate data."""
    rotf = RotationForestClassifier()

    # a univariate 3d array is squeezed to 2d
    X = rotf._check_X(np.random.random((10, 1, 100)))
    assert X.shape == (10, 100)

    # multivariate 3d and ragged inputs are rejected
    with pytest.raises(ValueError, match="not a time series"):
        rotf._check_X(np.random.random((10, 10, 100)))
    with pytest.raises(ValueError, match="not a time series"):
        rotf._check_X([[1, 2, 3], [4, 5], [6, 7, 8]])

    # constant attributes leave nothing to fit on
    X2 = np.zeros((10, 10))
    y = np.zeros(10)
    y[0:5] = 1

    with pytest.raises(ValueError, match="same value"):
        rotf.fit_predict(X2, y)


def test_rotf_tree_parameters():
    """Test exposed tree parameters reach the default decision trees."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")

    rotf = RotationForestClassifier(
        n_estimators=5,
        criterion="gini",
        max_depth=3,
        min_samples_leaf=2,
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    for tree in rotf.estimators_:
        assert tree.criterion == "gini"
        assert tree.max_depth == 3
        assert tree.min_samples_leaf == 2

    # the defaults leave the tree at entropy with no depth limit
    default = RotationForestClassifier(n_estimators=5, random_state=0)
    default.fit(X_train, y_train)
    assert default.estimators_[0].criterion == "entropy"
    assert default.estimators_[0].max_depth is None
