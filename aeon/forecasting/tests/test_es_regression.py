"""Test the ES normalised regression forecaster."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from aeon.forecasting import ESNormalisedRegressionForecaster


def _seasonal_series(n=96, m=12):
    t = np.arange(n)
    return 50 * (1 + 0.4 * np.sin(2 * np.pi * t / m)) * 1.01**t


def test_direct_forecast_shape_and_scale():
    """Direct forecasts have the requested length and stay on the data scale."""
    y = _seasonal_series()
    f = ESNormalisedRegressionForecaster(window=24, seasonal_period=12)
    preds = f.direct_forecast(y, 12)
    assert preds.shape == (12,)
    truth = _seasonal_series(108)[-12:]
    assert np.mean(np.abs(preds - truth) / truth) < 0.05


def test_iterative_forecast():
    """Iterative forecasting uses predict with the fitted ES parameters."""
    y = _seasonal_series()
    f = ESNormalisedRegressionForecaster(window=24, seasonal_period=12)
    preds = f.iterative_forecast(y, 6)
    assert preds.shape == (6,)
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize("log", [False, True])
def test_wraps_regressor(log):
    """The wrapped regressor is cloned, fitted and used for the forecast."""
    y = _seasonal_series()
    reg = RandomForestRegressor(n_estimators=10, random_state=0)
    f = ESNormalisedRegressionForecaster(
        window=12, horizon=3, regressor=reg, seasonal_period=12, log=log
    )
    f.fit(y)
    assert f.regressor_ is not reg
    assert not hasattr(reg, "estimators_")
    assert f.regressor_.n_features_in_ == 12
    assert f.forecast_ == pytest.approx(f.predict(y))


def test_window_shrinks_for_short_series():
    """The window is reduced so at least three training windows exist."""
    y = _seasonal_series(13)
    f = ESNormalisedRegressionForecaster(window=100, horizon=6, seasonal_period=12)
    f.fit(y)
    assert f.window_ == 13 - 6 - 2
    assert f.seasonal_period_ == 1
    with pytest.raises(ValueError, match="too short"):
        ESNormalisedRegressionForecaster(window=3, horizon=12).fit(y)


def test_fixed_parameters_are_used():
    """Supplied smoothing parameters are not re-estimated."""
    y = _seasonal_series()
    f = ESNormalisedRegressionForecaster(
        window=12, seasonal_period=12, alpha=0.3, gamma=0.1
    ).fit(y)
    assert f.alpha_ == 0.3 and f.gamma_ == 0.1
