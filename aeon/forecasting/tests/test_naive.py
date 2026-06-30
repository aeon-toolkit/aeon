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


@pytest.mark.parametrize("seasonal_period", [0, -1, 1.5, "2", True])
def test_naive_forecaster_seasonal_last_rejects_invalid_period(seasonal_period):
    """Test seasonal_last rejects invalid seasonal periods."""
    forecaster = NaiveForecaster(
        strategy="seasonal_last", seasonal_period=seasonal_period
    )

    with pytest.raises(ValueError, match="seasonal_period must be a positive integer"):
        forecaster.predict(np.arange(5.0))


def test_naive_forecaster_seasonal_last_rejects_period_longer_than_series():
    """Test seasonal_last rejects periods longer than the observed series."""
    forecaster = NaiveForecaster(
        strategy="seasonal_last", seasonal_period=10, horizon=10
    )

    with pytest.raises(ValueError, match="number of observations"):
        forecaster.predict(np.arange(5.0))


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
