"""Test the regression forecaster."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from aeon.forecasting import RegressionForecaster
from aeon.regression import DummyRegressor


class _LastFeatureRegressor(BaseEstimator):
    """Test regressor that predicts from the final feature column.

    Inherits :class:`sklearn.base.BaseEstimator` so that it satisfies
    sklearn's ``clone()`` contract (introduced into
    ``RegressionForecaster._fit`` by PR #3464 to prevent regressor
    mutation). ``BaseEstimator`` provides default ``get_params`` /
    ``set_params`` that are correct for this parameter-free class.
    """

    def fit(self, X, y):
        """Fit no state and return self."""
        return self

    def predict(self, X):
        """Return the final feature column."""
        return X[:, -1]


def test_regression_forecaster():
    """Test the regression forecaster.

    Test fit/predict and forecast are equivalent for sklearn and aeon regressors.
    Test invalid window handling.
    """
    y = np.random.rand(100)
    f = RegressionForecaster(window=10)
    p = f.forecast(y)
    f.fit(y)
    p2 = f.predict(y)
    assert p == p2
    f2 = RegressionForecaster(regressor=LinearRegression(), window=10)
    p2 = f2.forecast(y)
    assert p == p2
    f2 = RegressionForecaster(regressor=DummyRegressor(), window=10)
    f2.fit(y)
    f2.predict(y)

    with pytest.raises(ValueError):
        f = RegressionForecaster(window=-1)
        f.fit(y)
    with pytest.raises(ValueError):
        f = RegressionForecaster(window=101)
        f.fit(y)
    f = RegressionForecaster(window=10)
    f.fit(y)
    y_test = np.random.rand(5)
    with pytest.raises(ValueError):
        p = f.predict(y_test)


def test_regression_forecaster_with_exog():
    """Test the regression forecaster with exogenous variables."""
    n_samples = 100
    exog = np.random.rand(n_samples) * 10
    y = 2 * exog + np.random.rand(n_samples) * 0.1
    f = RegressionForecaster(window=10)

    # Test fit and predict with exog
    f.fit(y, exog=exog)
    p1 = f.forecast_
    assert isinstance(p1, float)

    # Test that exog variable have an impact
    exog_zeros = np.zeros(n_samples)
    f.fit(y, exog=exog_zeros)
    p2 = f.forecast_
    assert p1 != p2

    # Test that forecast method works and is equivalent to fit+predict
    y_new = np.arange(50, 150)
    exog_new = np.arange(50, 150) * 2

    # Manual fit + predict
    f.fit(y=y_new, exog=exog_new)
    p_manual = f.predict(y_new, exog=exog_new[-1:])

    # forecast() method
    p_forecast = f.forecast(y=y_new, exog=exog_new)
    assert p_manual == pytest.approx(p_forecast)

    # Test with multivariate exog
    exog_m = np.column_stack([exog, exog_zeros])
    p1 = f.forecast(y, exog_m)
    f.fit(y, exog_m)
    p2 = f.predict(y, exog_m[-1:])
    assert p1 == p2
    y = np.random.random((1, 100))
    exog = np.random.random((1, 100))
    f._fit(y, exog)


def test_regression_forecaster_predict_uses_target_time_exog():
    """RegressionForecaster predict should accept a single target exog row."""
    y = np.arange(20, dtype=float)
    exog = np.arange(20, dtype=float)
    f = RegressionForecaster(window=4, regressor=_LastFeatureRegressor())
    f.fit(y, exog=exog)

    assert f.predict(y, exog=np.array([123.0])) == 123.0


def test_regression_forecaster_iterative_forecast_with_exog():
    """RegressionForecaster should pass one future exog row per forecast step."""
    y = np.arange(20, dtype=float)
    exog = np.arange(20, dtype=float)
    future_exog = np.array([101.0, 102.0, 103.0])
    f = RegressionForecaster(window=4, regressor=_LastFeatureRegressor())

    preds = f.iterative_forecast(
        y,
        prediction_horizon=3,
        exog=exog,
        future_exog=future_exog,
    )

    np.testing.assert_allclose(preds, future_exog)


def test_regression_forecaster_iterative_forecast_uses_changed_future_exog():
    """Changing future_exog should change iterative forecasts."""
    y = np.arange(20, dtype=float)
    exog = np.arange(20, dtype=float)
    f = RegressionForecaster(window=4, regressor=_LastFeatureRegressor())

    preds_a = f.iterative_forecast(
        y,
        prediction_horizon=2,
        exog=exog,
        future_exog=np.array([10.0, 20.0]),
    )
    preds_b = f.iterative_forecast(
        y,
        prediction_horizon=2,
        exog=exog,
        future_exog=np.array([30.0, 40.0]),
    )

    assert not np.array_equal(preds_a, preds_b)


def test_regression_forecaster_predict_rejects_full_exog_history():
    """RegressionForecaster predict should reject full exog history."""
    y = np.arange(20, dtype=float)
    exog = np.arange(20, dtype=float)
    f = RegressionForecaster(window=4).fit(y, exog=exog)

    with pytest.raises(ValueError, match="single target-time row"):
        f.predict(y, exog=exog)


def test_regression_forecaster_with_exog_errors():
    """Test errors in regression forecaster with exogenous variables."""
    y = np.random.rand(100)
    exog_short = np.random.rand(99)
    f = RegressionForecaster(window=10)

    # Test for unequal length series in fit
    with pytest.raises(ValueError, match="one row per time point"):
        f.fit(y, exog=exog_short)
    # Test for fit/predict mismatches in shape

    # If exog in fit, must have them in predict
    exog_train = np.array(np.random.rand(100))
    with pytest.raises(ValueError, match="predict passed no exogenous variables"):
        f.fit(y, exog=exog_train)
        f.predict(y)
    exog_test = np.array([np.random.rand(10), np.random.rand(10)])
    f.fit(y, exog=exog_train)
    with pytest.raises(ValueError, match="single target-time row"):
        f.predict(y, exog_test)
    exog_short = np.random.rand(5)
    with pytest.raises(ValueError, match="single target-time row"):
        f.predict(y, exog_short)
    with pytest.raises(ValueError, match="predict passed no exogenous variables"):
        f.predict(y)
    with pytest.raises(ValueError, match="must be greater than or equal to 1"):
        f.direct_forecast(y, prediction_horizon=0)


def test_regressor_cloned_not_mutated():
    """Test RegressionForecaster clones the regressor instead of mutating it."""
    import numpy as np
    from sklearn.linear_model import Ridge

    from aeon.forecasting._regression import RegressionForecaster

    reg = Ridge()
    y1 = np.arange(20.0)
    y2 = np.arange(20.0)[::-1]

    f1 = RegressionForecaster(window=5, regressor=reg)
    f2 = RegressionForecaster(window=5, regressor=reg)

    f1.fit(y1)
    f2.fit(y2)

    # Original regressor should not be the fitted regressor_
    assert reg is not f1.regressor_
    assert reg is not f2.regressor_
    # Two forecasters with same regressor should get independent clones
    assert f1.regressor_ is not f2.regressor_
