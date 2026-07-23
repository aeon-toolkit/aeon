"""Test Naive Forecaster."""

import numpy as np
import pytest

from aeon.forecasting import NaiveForecaster


def test_naive_forecaster_last_strategy():
    """Test NaiveForecaster with 'last' strategy."""
    sample_data = np.array([10, 20, 30, 40, 50])
    forecaster = NaiveForecaster(strategy="last", horizon=3)
    predictions = forecaster.predict(sample_data)
    expected = 50
    np.testing.assert_array_equal(predictions, expected)


def test_naive_forecaster_mean_strategy():
    """Test NaiveForecaster with 'mean' strategy."""
    sample_data = np.array([10, 20, 30, 40, 50])
    forecaster = NaiveForecaster(strategy="mean", horizon=2)
    predictions = forecaster.predict(sample_data)
    expected = 30  # Mean of [10, 20, 30, 40, 50] is 30
    np.testing.assert_array_equal(predictions, expected)


def test_naive_forecaster_seasonal_last_strategy():
    """Test NaiveForecaster with 'seasonal_last' strategy."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    # Last season is [6, 7, 8] for seasonal_period = 3
    forecaster = NaiveForecaster(strategy="seasonal_last", seasonal_period=3, horizon=4)
    pred = forecaster.forecast(data)
    forecaster.fit(data)
    pred = forecaster.predict(data)
    expected = 6  # predicts the 1-st element of the last season.
    np.testing.assert_array_equal(pred, expected)

    # Test horizon within the season length
    forecaster = NaiveForecaster(strategy="seasonal_last", seasonal_period=3, horizon=2)
    pred = forecaster.forecast(data)
    forecaster.fit(data)
    pred = forecaster.predict(data)
    expected = 7  # predicts the 2-nd element of the last season.
    np.testing.assert_array_equal(pred, expected)

    # Test horizon wrapping around to a new season
    forecaster = NaiveForecaster(strategy="seasonal_last", seasonal_period=3, horizon=7)
    pred = forecaster.forecast(data)
    forecaster.fit(data)
    pred = forecaster.predict(data)
    expected = 6  # predicts the 1-st element of the last season.
    np.testing.assert_array_equal(pred, expected)

    # Last season is now [5, 6, 7, 8] with seasonal_period = 4
    forecaster = NaiveForecaster(strategy="seasonal_last", seasonal_period=4, horizon=6)
    pred = forecaster.forecast(data)
    forecaster.fit(data)
    pred = forecaster.predict(data)
    expected = 6  # predicts the 2nd element of the new last season.
    np.testing.assert_array_equal(pred, expected)


def test_naive_forecaster_drift_strategy():
    """Test 'drift' extrapolates the first-to-last trend line.

    For a perfectly linear series the drift forecast lies exactly on the line, so
    the h-step-ahead value is the last observation plus h times the constant slope.
    """
    slope = 2.0
    y = np.arange(2, 12, 2, dtype=float)  # [2, 4, 6, 8, 10], slope 2 per step
    last = y[-1]

    for horizon in (1, 3, 5):
        f = NaiveForecaster(strategy="drift", horizon=horizon)
        np.testing.assert_allclose(f.forecast(y), last + horizon * slope)


def test_naive_forecaster_drift_direct_matches_iterative():
    """Drift gives the same next-N line via direct and iterative forecasting.

    Appending an on-line prediction leaves the first-to-last slope unchanged, so
    the recursive (iterative) forecast must reproduce the direct straight-line
    extrapolation. Uses the default horizon=1 required by iterative forecasting.
    """
    slope = 3.0
    y = np.arange(0, 15, 3, dtype=float)  # [0, 3, 6, 9, 12], slope 3 per step
    n_ahead = 4
    f = NaiveForecaster(strategy="drift")

    direct = f.direct_forecast(y, n_ahead)
    iterative = f.iterative_forecast(y, n_ahead)

    np.testing.assert_allclose(direct, iterative)
    expected = y[-1] + slope * np.arange(1, n_ahead + 1)
    np.testing.assert_allclose(iterative, expected)


def test_naive_forecaster_drift_requires_two_observations():
    """Drift cannot estimate a trend from a single point and must raise."""
    f = NaiveForecaster(strategy="drift")
    with pytest.raises(ValueError, match="at least two observations"):
        f.forecast(np.array([5.0]))


def test_predict():
    """Test different input for private predict."""
    forecaster = NaiveForecaster(strategy="mean")
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    pred = forecaster._predict(y)
    np.testing.assert_allclose(pred, 4.5)
    assert isinstance(pred, float)
    forecaster = NaiveForecaster(strategy="seasonal_last", seasonal_period=2)
    pred = forecaster._predict(y)
    assert pred == 7.0
    forecaster = NaiveForecaster(strategy="FOOBAR")
    with pytest.raises(ValueError, match="Unknown strategy"):
        forecaster._predict(y)
