"""Test SETAR-Tree forecaster."""

import numpy as np

from aeon.forecasting.machine_learning._setartree import SETARTree


def test_constant_series():
    """Test forecasting on a constant series (scalar prediction)."""
    y = np.ones(20)
    f = SETARTree(lag=2)
    f.fit(y)
    pred = f.predict(y)
    assert np.isscalar(pred), "Prediction should be a scalar for univariate series"
    assert np.isclose(pred, 1.0), f"Prediction {pred} not close to 1.0"


def test_linear_series():
    """Test forecasting on a linear series (scalar prediction)."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2)
    f.fit(y)
    pred = f.predict(y)
    assert np.isscalar(pred), "Prediction should be a scalar for univariate series"
    assert np.isclose(pred, 21.0, atol=0.01), f"Prediction {pred} not close to 21.0"


def test_iterative_forecast_linear():
    """Test iterative forecasting on a linear series (1-D array of length H)."""
    y = np.arange(1, 11.0)
    f = SETARTree(lag=2)
    preds = f.iterative_forecast(y, 3)
    expected = np.arange(11.0, 14.0)
    assert preds.shape == (
        3,
    ), f"iterative_forecast should return shape (H,), got {preds.shape}"
    assert np.allclose(
        preds, expected, atol=0.01
    ), f"Predictions {preds} not close to {expected}"


def test_fixed_lag():
    """Test the fixed_lag parameter."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2, fixed_lag=True, external_lag=1)
    f.fit(y)
    pred = f.predict(y)
    assert np.isclose(
        pred, 21.0, atol=0.01
    ), f"Fixed lag prediction {pred} not close to 21.0"


def test_different_stopping_criteria():
    """Test different stopping criteria parameters."""
    y = np.arange(1, 21.0)
    for criteria in ["lin_test", "error_imp", "both"]:
        f = SETARTree(lag=2, stopping_criteria=criteria)
        f.fit(y)
        pred = f.predict(y)
        assert np.isclose(
            pred, 21.0, atol=0.01
        ), f"Prediction with {criteria} {pred} not close to 21.0"


def test_forecast_linear_sets_attribute():
    """Test forecast() returns scalar next value and sets forecast_ attribute."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2)
    out = f.forecast(y)
    assert np.isscalar(out), "forecast() should return a scalar for univariate series"
    assert np.isclose(out, 21.0, atol=0.01), f"Forecast {out} not close to 21.0"
    assert hasattr(f, "forecast_"), "forecast_ attribute should be set by _forecast()"
    assert np.isclose(f.forecast_, out), "forecast_ should equal the returned forecast"
