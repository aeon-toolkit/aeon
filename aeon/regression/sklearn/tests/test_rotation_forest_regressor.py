"""Rotation Forest test code."""

import numpy as np
from sklearn.metrics import mean_squared_error

from aeon.datasets import load_covid_3month
from aeon.regression.sklearn import RotationForestRegressor


def test_rotf_output():
    """Test of RotF contracting and train estimate on test data."""
    X_train, y_train = load_covid_3month(split="train", return_type="numpy2d")
    X_test, y_test = load_covid_3month(split="test", return_type="numpy2d")

    rotf = RotationForestRegressor(
        n_estimators=10,
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    expected = [
        0.02942543,
        0.02087061,
        0.01973804,
        0.06120395,
        0.06659284,
        0.01891775,
        0.04199085,
        0.01,
        0.03088715,
        0.05546291,
        0.01987163,
        0.03721117,
        0.0172066,
        0.02747714,
        0.01639872,
    ]

    np.testing.assert_array_almost_equal(expected, rotf.predict(X_test[:15]), decimal=4)


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
    """Test of RotF fit_predict on testing data."""
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
