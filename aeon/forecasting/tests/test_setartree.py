"""Test SETAR-Tree forecaster."""

import numpy as np

from aeon.forecasting._setartree import SETARTree


def test_short_series():
    """Test handling of series shorter than the lag."""
    y = np.random.random(5)
    f = SETARTree(lag=10)
    try:
        f.fit(y)
        raise AssertionError("No ValueError raised for short series")
    except ValueError as e:
        assert "insufficient data" in str(e).lower(), "Unexpected error message"


def test_constant_series():
    """Test forecasting on a constant series."""
    y = np.ones(20)
    f = SETARTree(lag=2)
    f.fit(y)
    pred = f.predict(y)
    assert np.isclose(pred, 1), f"Prediction {pred} not close to 1"


def test_linear_series():
    """Test forecasting on a linear series."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2)
    f.fit(y)
    pred = f.predict(y)
    assert np.isclose(pred, 21, atol=0.01), f"Prediction {pred} not close to 21"


def test_iterative_forecast_linear():
    """Test iterative forecasting on a linear series."""
    y = np.arange(1, 11.0)
    f = SETARTree(lag=2)
    preds = f.iterative_forecast(y, 3)
    expected = np.arange(11, 14.0)
    assert np.allclose(
        preds, expected, atol=0.01
    ), f"Predictions {preds} not close to {expected}"


def test_scale_option():
    """Test the scale parameter functionality."""
    y = np.arange(1, 21.0) * 100  # Large values to test scaling
    f = SETARTree(lag=2, scale=True)
    f.fit(y)
    pred = f.predict(y)
    assert np.isclose(pred, 2100, atol=1), f"Scaled prediction {pred} not close to 2100"


def test_fixed_lag():
    """Test the fixed_lag parameter."""
    y = np.arange(1, 21.0)
    f = SETARTree(lag=2, fixed_lag=True, external_lag=1)
    f.fit(y)
    pred = f.predict(y)
    assert np.isclose(
        pred, 21, atol=0.01
    ), f"Fixed lag prediction {pred} not close to 21"


def test_different_stopping_criteria():
    """Test different stopping criteria parameters."""
    y = np.arange(1, 21.0)
    for criteria in ["lin_test", "error_imp", "both"]:
        f = SETARTree(lag=2, stopping_criteria=criteria)
        f.fit(y)
        pred = f.predict(y)
        assert np.isclose(
            pred, 21, atol=0.01
        ), f"Prediction with {criteria} {pred} not close to 21"
