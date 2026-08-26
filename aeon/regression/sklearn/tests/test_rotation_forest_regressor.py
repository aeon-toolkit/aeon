"""Rotation Forest test code."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from aeon.datasets import load_covid_3month
from aeon.regression.sklearn import RotationForestRegressor


def test_rotf_output():
    """Test RotF error matches the expected value on the covid 3 month data.

    Unlike the classifier test, an aggregate error is checked rather than
    per-case predictions: scikit-learn 1.8 changed decision-tree handling of
    almost constant features (present in this dataset), which shifts individual
    predictions between supported scikit-learn versions (see the class
    docstring). The error is stable across versions.
    """
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")
    X_test, y_test = load_covid_3month(split="test", return_type="numpy2d")

    rotf = RotationForestRegressor(
        n_estimators=10,
        base_estimator=DecisionTreeRegressor(max_depth=3),
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    y_pred = rotf.predict(X_test)
    assert y_pred.shape == y_test.shape

    mse = mean_squared_error(y_test, y_pred)
    np.testing.assert_almost_equal(mse, 0.0019, decimal=4)


def test_contracted_rotf():
    """Test contracted RotF stays within the contract and keeps its error."""
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")
    X_test, y_test = load_covid_3month(split="test", return_type="numpy2d")

    contract_max_n_estimators = 5

    rotf = RotationForestRegressor(
        time_limit_in_minutes=5,
        contract_max_n_estimators=contract_max_n_estimators,
        random_state=0,
    )
    rotf.fit(X_train, y_train)
    assert 0 < len(rotf.estimators_) <= contract_max_n_estimators

    y_pred = rotf.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_test.shape

    mse = mean_squared_error(y_test, y_pred)
    np.testing.assert_almost_equal(mse, 0.0021, decimal=4)


def test_rotf_fit_predict():
    """Test RotF fit_predict returns train prediction estimates."""
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")
    n_estimators = 5

    rotf = RotationForestRegressor(
        n_estimators=n_estimators,
        random_state=0,
    )

    y_pred = rotf.fit_predict(X_train, y_train)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_train.shape
    assert len(rotf.estimators_) == n_estimators
    assert rotf._is_fitted

    y_pred = rotf.predict(X_train)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_train.shape
