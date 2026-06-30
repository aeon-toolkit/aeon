"""Test base forecaster."""

import numpy as np
import pandas as pd
import pytest

from aeon.forecasting import NaiveForecaster, RegressionForecaster
from aeon.forecasting.base import BaseForecaster


class _FitCountingRegressionForecaster(RegressionForecaster):
    """RegressionForecaster test double that counts internal fit calls."""

    def __init__(self, window=4):
        self.fit_calls_ = 0
        super().__init__(window=window)

    def _fit(self, y, exog=None):
        self.fit_calls_ += 1
        return super()._fit(y, exog=exog)


def test_base_forecaster():
    """Test base forecaster functionality."""
    f = NaiveForecaster()
    y = np.random.rand(50)
    f.fit(y)
    p1 = f.predict(y)
    assert p1 == y[-1]
    p2 = f.forecast(y)
    p3 = f._forecast(y, None)
    assert p2 == p1
    assert p3 == p2
    with pytest.raises(ValueError, match="Exogenous variables passed"):
        f.forecast(y, exog=y)


def test_naive_seasonal_last_validates_seasonal_period():
    """seasonal_last must raise a clear error for an invalid seasonal_period (gh-3576)."""
    y = np.arange(20, dtype=float)
    for bad in (0, -1, None):
        f = NaiveForecaster(strategy="seasonal_last", seasonal_period=bad)
        f.fit(y)
        with pytest.raises(ValueError, match="seasonal_period"):
            f.predict(y)
    # a seasonal_period larger than the series is also rejected, not an IndexError
    short = np.arange(10, dtype=float)
    f = NaiveForecaster(strategy="seasonal_last", seasonal_period=50)
    f.fit(short)
    with pytest.raises(ValueError, match="cannot exceed"):
        f.predict(short)


def test_convert_y():
    """Test y conversion in forecasting base."""
    f = NaiveForecaster()
    y = np.random.rand(50)
    with pytest.raises(ValueError, match="Input axis should be 0 or 1"):
        f._convert_y(y, axis=2)
    y2 = f._convert_y(pd.Series(y), axis=0)
    assert isinstance(y2, np.ndarray)
    y = np.random.random((100, 2))
    y2 = f._convert_y(y, axis=0)
    assert y2.shape == (2, 100)
    f.set_tags(**{"y_inner_type": "pd.DataFrame"})
    y2 = f._convert_y(y, axis=0)
    assert isinstance(y2, pd.DataFrame)
    y2 = f._convert_y(y, axis=1)
    assert isinstance(y2, pd.DataFrame)
    f.set_tags(**{"y_inner_type": "pd.Series"})
    with pytest.raises(ValueError, match="Unsupported inner type"):
        f._convert_y(y, axis=1)
    with pytest.raises(ValueError, match="must be greater than or equal to 1"):
        f.direct_forecast(y, prediction_horizon=0)


def test_direct_forecast():
    """Test direct forecasting."""
    y = np.random.rand(50)
    f = RegressionForecaster(window=10)
    # Direct should be the same as setting horizon manually.
    preds = f.direct_forecast(y, prediction_horizon=10)
    assert isinstance(preds, np.ndarray) and len(preds) == 10
    for i in range(0, 10):
        f = RegressionForecaster(window=10, horizon=i + 1)
        p = f.forecast(y)
        assert p == preds[i]


def test_iterative_forecast():
    """Test terativeforecasting."""
    y = np.random.rand(50)
    f = RegressionForecaster(window=4)
    preds = f.iterative_forecast(y, prediction_horizon=10)
    assert isinstance(preds, np.ndarray) and len(preds) == 10
    f.fit(y)
    for i in range(0, 10):
        p = f.predict(y)
        assert p == preds[i]
        y = np.append(y, p)


def test_iterative_forecast_fits_once():
    """Test iterative forecasting calls fit once."""
    y = np.random.rand(50)
    f = _FitCountingRegressionForecaster(window=4)

    preds = f.iterative_forecast(y, prediction_horizon=10)

    assert isinstance(preds, np.ndarray) and len(preds) == 10
    assert f.fit_calls_ == 1


def test_iterative_forecast_rejects_future_exog_without_exog():
    """Test future_exog without exog is rejected."""
    y = np.random.rand(50)
    future_exog = np.random.rand(3, 2)
    f = RegressionForecaster(window=4)

    with pytest.raises(ValueError, match="provided together"):
        f.iterative_forecast(y, prediction_horizon=3, future_exog=future_exog)


def test_iterative_forecast_rejects_exog_without_future_exog():
    """Test exog without future_exog is rejected."""
    y = np.random.rand(50)
    exog = np.random.rand(50, 2)
    f = RegressionForecaster(window=4)

    with pytest.raises(ValueError, match="provided together"):
        f.iterative_forecast(y, prediction_horizon=3, exog=exog)


def test_iterative_forecast_rejects_wrong_future_exog_length():
    """Test future_exog length must match the forecast horizon."""
    y = np.random.rand(50)
    exog = np.random.rand(50, 2)
    future_exog = np.random.rand(2, 2)
    f = RegressionForecaster(window=4)

    with pytest.raises(ValueError, match="forecast horizon step"):
        f.iterative_forecast(
            y, prediction_horizon=3, exog=exog, future_exog=future_exog
        )


def test_iterative_forecast_rejects_exog_feature_mismatch():
    """Test exog and future_exog must have the same feature count."""
    y = np.random.rand(50)
    exog = np.random.rand(50, 2)
    future_exog = np.random.rand(3, 3)
    f = RegressionForecaster(window=4)

    with pytest.raises(ValueError, match="same number of features"):
        f.iterative_forecast(
            y, prediction_horizon=3, exog=exog, future_exog=future_exog
        )


def test_output_equivalence():
    """Test output same for one ahead forecast."""
    y = np.random.rand(50)
    f = RegressionForecaster(window=4)
    p1 = f.forecast(y)
    p2 = f.fit(y).predict(y)
    p3 = f.iterative_forecast(y, 1)
    p4 = f.direct_forecast(y, 1)
    assert np.allclose(p1, p2, p3[0], p4[0])


def test_direct_forecast_with_exog():
    """Test direct forecasting with exogenous variables."""
    y = np.arange(50)
    exog = np.arange(50) * 2
    f = RegressionForecaster(window=10)

    preds = f.direct_forecast(y, prediction_horizon=10, exog=exog)
    assert isinstance(preds, np.ndarray) and len(preds) == 10

    # Check that predictions are different from when no exog is used
    preds_no_exog = f.direct_forecast(y, prediction_horizon=10)
    assert not np.array_equal(preds, preds_no_exog)


def test_fit_is_empty():
    """Test empty fit."""

    class _EmptyFit(BaseForecaster):
        _tags = {"fit_is_empty": True}

        def _fit(self, y):
            return self

        def _predict(self, y):
            return 0

    dummy = _EmptyFit(horizon=1, axis=1)
    y = np.arange(50)
    dummy.fit(y)
    assert dummy.is_fitted
