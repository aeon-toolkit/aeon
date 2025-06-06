"""Test Naive Forecaster."""

import numpy as np

from aeon.forecasting import NaiveForecaster


def test_naive_forecaster_last_strategy():
    """Test NaiveForecaster with 'last' strategy."""
    sample_data = np.array([10, 20, 30, 40, 50])
    forecaster = NaiveForecaster(strategy="last", horizon=3)
    forecaster.fit(sample_data)
    predictions = forecaster.predict()
    expected = np.array([50, 50, 50])
    np.testing.assert_array_equal(predictions, expected)


def test_naive_forecaster_mean_strategy():
    """Test NaiveForecaster with 'mean' strategy."""
    sample_data = np.array([10, 20, 30, 40, 50])
    forecaster = NaiveForecaster(strategy="mean", horizon=2)
    forecaster.fit(sample_data)
    predictions = forecaster.predict()
    expected = np.array(30)  # Mean of [10, 20, 30, 40, 50] is 30
    np.testing.assert_array_equal(predictions, expected)


def test_naive_forecaster_seasonal_last_strategy():
    """Test NaiveForecaster with 'seasonal_last' strategy."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # More data for seasonality
    forecaster = NaiveForecaster(strategy="seasonal_last", seasonal_period=3, horizon=4)
    forecaster.fit(data)  # Last season is [6, 7, 8]
    predictions = forecaster.predict()
    expected = np.array(6)
    np.testing.assert_array_equal(predictions, expected)


def test_naive_forecaster_forecast_method():
    """Test the _forecast method (direct forecasting without prior fit)."""
    sample_data = np.array([10, 20, 30, 40, 50])
    forecaster = NaiveForecaster(strategy="last", horizon=2)
    predictions = forecaster._forecast(sample_data)  # Using _forecast directly
    expected = np.array([50, 50])
    np.testing.assert_array_equal(predictions, expected)

    forecaster_mean = NaiveForecaster(strategy="mean", horizon=2)
    predictions_mean = forecaster_mean._forecast(sample_data)
    expected_mean = np.array([30, 30])
    np.testing.assert_array_equal(predictions_mean, expected_mean)
