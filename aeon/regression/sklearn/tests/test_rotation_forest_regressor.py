"""Rotation Forest test code."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from aeon.datasets import load_covid_3month
from aeon.regression.sklearn import RotationForestRegressor


def test_rotf_output():
    """Test RotF predictions match expected values on the covid 3 month data."""
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")
    X_test, y_test = load_covid_3month(split="test", return_type="numpy2d")

    rotf = RotationForestRegressor(
        n_estimators=10,
        base_estimator=DecisionTreeRegressor(max_depth=3),
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    expected = [
        0.03020,
        0.02127,
        0.02160,
        0.04964,
        0.06710,
        0.03825,
        0.02254,
        0.04176,
        0.02631,
        0.04706,
        0.02886,
        0.02560,
        0.02607,
        0.03558,
        0.02300,
    ]
    np.testing.assert_array_almost_equal(expected, rotf.predict(X_test[:15]), decimal=4)


def test_rotf_pca_solver_is_noop():
    """Test pca_solver is retained for compatibility but has no effect."""
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")
    X_test, _ = load_covid_3month(split="test", return_type="numpy2d")

    rotf_full = RotationForestRegressor(
        n_estimators=10,
        pca_solver="full",
        random_state=0,
    )
    rotf_randomized = RotationForestRegressor(
        n_estimators=10,
        pca_solver="randomized",
        random_state=0,
    )

    rotf_full.fit(X_train, y_train)
    rotf_randomized.fit(X_train, y_train)

    np.testing.assert_array_equal(
        rotf_full.predict(X_test[:15]),
        rotf_randomized.predict(X_test[:15]),
    )


def test_contracted_rotf():
    """Test of RotF contracting on testing data."""
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")
    X_test, y_test = load_covid_3month(split="test", return_type="numpy2d")

    rotf = RotationForestRegressor(
        time_limit_in_minutes=5,
        contract_max_n_estimators=5,
        random_state=0,
    )

    rotf.fit(X_train, y_train)
    assert len(rotf.estimators_) > 0

    y_pred = rotf.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(y_test)

    mse = mean_squared_error(y_test, y_pred)
    np.testing.assert_almost_equal(mse, 0.002, decimal=4)


def test_rotf_fit_predict():
    """Test RotF fit_predict returns train prediction estimates."""
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")

    rotf = RotationForestRegressor(
        n_estimators=5,
        random_state=0,
    )

    y_pred = rotf.fit_predict(X_train, y_train)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(y_train)
    assert len(rotf.estimators_) > 0
    assert rotf._is_fitted

    y_pred = rotf.predict(X_train)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(y_train)


def test_rotf_input():
    """Test RotF rejects unsupported input shapes and degenerate data."""
    rotf = RotationForestRegressor()

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
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")

    rotf = RotationForestRegressor(
        n_estimators=5,
        splitter="random",
        max_depth=3,
        min_samples_leaf=2,
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    for tree in rotf.estimators_:
        assert tree.splitter == "random"
        assert tree.max_depth == 3
        assert tree.min_samples_leaf == 2

    # the defaults leave the tree at squared_error with the best splitter
    default = RotationForestRegressor(n_estimators=5, random_state=0)
    default.fit(X_train, y_train)
    assert default.estimators_[0].criterion == "squared_error"
    assert default.estimators_[0].splitter == "best"
