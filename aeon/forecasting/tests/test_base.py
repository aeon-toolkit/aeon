"""Test base forecaster."""

import numpy as np
import pandas as pd
import pytest

from aeon.forecasting import NaiveForecaster, RegressionForecaster
from aeon.forecasting.base import (
    BaseForecaster,
    DirectForecastingMixin,
    SeriesToSeriesForecastingMixin,
)


class _FitCountingRegressionForecaster(RegressionForecaster):
    """RegressionForecaster test double that counts internal fit calls."""

    def __init__(self, window=4):
        self.fit_calls_ = 0
        super().__init__(window=window)

    def _fit(self, y, exog=None):
        self.fit_calls_ += 1
        return super()._fit(y, exog=exog)


class _NoHorizonForecaster(BaseForecaster, DirectForecastingMixin):
    """Forecaster test double without multi-horizon capability."""

    _tags = {"capability:horizon": False, "fit_is_empty": True}

    def _predict(self, y, exog=None):
        """Predict a constant value."""
        return 0.0


class _SeriesToSeriesForecaster(BaseForecaster, SeriesToSeriesForecastingMixin):
    """Series-to-series forecasting test double."""

    _tags = {"fit_is_empty": True}

    def _predict(self, y, exog=None):
        """Predict a constant value."""
        return 0.0

    def _series_to_series_forecast(self, y, prediction_horizon, exog=None):
        """Return a fixed-length forecast."""
        return np.arange(prediction_horizon, dtype=float)


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
    """seasonal_last must raise a clear error for an invalid period (gh-3576)."""
    y = np.arange(20, dtype=float)
    # True is included because bool is a subclass of int (must be rejected)
    for bad in (0, -1, None, True):
        f = NaiveForecaster(strategy="seasonal_last", seasonal_period=bad)
        f.fit(y)
        with pytest.raises(ValueError, match="seasonal_period"):
            f.predict(y)
    # a seasonal_period larger than the series is also rejected, not an IndexError.
    # horizon=15 would have overflowed the effective period under the old code.
    short = np.arange(10, dtype=float)
    f = NaiveForecaster(strategy="seasonal_last", seasonal_period=50, horizon=15)
    f.fit(short)
    with pytest.raises(ValueError, match="cannot exceed"):
        f.predict(short)


def test_base_forecaster_rejects_unsupported_horizon():
    """Test horizon validation for forecasters without horizon capability."""
    y = np.arange(10, dtype=float)
    f = _NoHorizonForecaster(horizon=2, axis=1)

    with pytest.raises(ValueError, match="cannot handle a horizon greater than 1"):
        f.forecast(y)


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


def test_direct_forecast_rejects_forecaster_without_horizon_capability():
    """Test direct forecasting requires multi-horizon capability."""
    y = np.arange(10, dtype=float)
    f = _NoHorizonForecaster(horizon=1, axis=1)

    with pytest.raises(ValueError, match="cannot be used with the direct strategy"):
        f.direct_forecast(y, prediction_horizon=2)


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


@pytest.mark.parametrize(
    "kwargs, exception, match",
    [
        # prediction_horizon must be a genuine integer (bool/float/str rejected)
        (dict(prediction_horizon=True), TypeError, "must be an integer"),
        (dict(prediction_horizon=1.5), TypeError, "must be an integer"),
        (dict(prediction_horizon="2"), TypeError, "must be an integer"),
        # exog and future_exog must be supplied together (both directions)
        (
            dict(prediction_horizon=3, future_exog=np.ones((3, 2))),
            ValueError,
            "provided together",
        ),
        (
            dict(prediction_horizon=3, exog=np.ones((50, 2))),
            ValueError,
            "provided together",
        ),
        # exog must have one row per time point in y
        (
            dict(
                prediction_horizon=3,
                exog=np.ones((49, 2)),
                future_exog=np.ones((3, 2)),
            ),
            ValueError,
            "one row per time point",
        ),
        # exog / future_exog must be 1D or 2D
        (
            dict(
                prediction_horizon=3,
                exog=np.ones((50, 2, 1)),
                future_exog=np.ones((3, 2)),
            ),
            ValueError,
            "exog must be a 1D or 2D",
        ),
        (
            dict(
                prediction_horizon=3,
                exog=np.ones((50, 2)),
                future_exog=np.ones((3, 2, 1)),
            ),
            ValueError,
            "future_exog must be a 1D or 2D",
        ),
        # future_exog must have one row per forecast horizon step
        (
            dict(
                prediction_horizon=3,
                exog=np.ones((50, 2)),
                future_exog=np.ones((2, 2)),
            ),
            ValueError,
            "forecast horizon step",
        ),
        # exog and future_exog must share the same feature count
        (
            dict(
                prediction_horizon=3,
                exog=np.ones((50, 2)),
                future_exog=np.ones((3, 3)),
            ),
            ValueError,
            "same number of features",
        ),
    ],
)
def test_iterative_forecast_input_validation(kwargs, exception, match):
    """Test iterative forecasting validates prediction_horizon and exog inputs."""
    y = np.arange(50, dtype=float)
    f = RegressionForecaster(window=4)

    with pytest.raises(exception, match=match):
        f.iterative_forecast(y, **kwargs)


def test_iterative_forecast_rejects_scalar_y():
    """Test iterative forecasting rejects scalar y."""
    f = RegressionForecaster(window=4)

    with pytest.raises(ValueError, match="at least one-dimensional"):
        f.iterative_forecast(np.array(1.0), prediction_horizon=1)


@pytest.mark.parametrize(
    "forecaster",
    [
        RegressionForecaster(window=4, horizon=3),
        NaiveForecaster(strategy="seasonal_last", seasonal_period=3, horizon=2),
    ],
)
def test_iterative_forecast_requires_unit_horizon(forecaster):
    """iterative_forecast rejects a forecaster configured with horizon > 1.

    Iterative forecasting recursively feeds each prediction back as the next
    observation, which is only defined when the base forecaster predicts one step
    ahead. A horizon greater than 1 must raise rather than silently produce an
    ill-defined multi-step sequence.
    """
    y = np.arange(1, 31, dtype=float)
    with pytest.raises(ValueError, match="iterative_forecast requires"):
        forecaster.iterative_forecast(y, prediction_horizon=5)


def test_iterative_matches_direct_for_seasonal_naive():
    """Seasonal-naive gives the same next-N forecast via direct and iterative.

    Both strategies must reproduce the model's repeated last season. This pins the
    recursion phase: iterative feeds one-step predictions back and must walk the
    season forward in step with direct, which refits per horizon.
    """
    seasonal_period = 3
    y = np.arange(1, 13, dtype=float)  # last season is [10, 11, 12]
    f = NaiveForecaster(strategy="seasonal_last", seasonal_period=seasonal_period)
    horizon_steps = 2 * seasonal_period
    direct = f.direct_forecast(y, horizon_steps)
    iterative = f.iterative_forecast(y, horizon_steps)
    np.testing.assert_array_equal(direct, iterative)
    expected = np.tile(y[-seasonal_period:], horizon_steps // seasonal_period)
    np.testing.assert_array_equal(iterative, expected)


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
    y = np.arange(50)
    forecaster = NaiveForecaster()

    forecaster.fit(y)

    assert forecaster.is_fitted


def test_series_to_series_forecast():
    """Test series-to-series forecasting validation and dispatch."""
    y = np.arange(10, dtype=float)
    f = _SeriesToSeriesForecaster(horizon=1, axis=1)

    np.testing.assert_array_equal(f.series_to_series_forecast(y, 3), [0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="must be greater than or equal to 1"):
        f.series_to_series_forecast(y, 0)


def test_convert_y_list_inner_type_and_dataframe():
    """_convert_y handles list y_inner_type tags and DataFrame conversion."""
    f = NaiveForecaster()
    f.set_tags(**{"y_inner_type": ["np.ndarray"]})
    out = f._convert_y(pd.Series(np.arange(5.0)), axis=1)
    assert isinstance(out, np.ndarray)

    f2 = NaiveForecaster()
    f2.set_tags(**{"y_inner_type": "pd.DataFrame"})
    out2 = f2._convert_y(np.arange(5.0), axis=1)
    assert isinstance(out2, pd.DataFrame)
    # 1D input with axis=1 is transposed to a single row
    assert out2.shape == (1, 5)
