"""Test functions for SeriesToSeriesForecastingMixin and DemoSeriesForecaster."""

import numpy as np
import pytest

from aeon.forecasting.base import SeriesToSeriesForecastingMixin
from aeon.forecasting.deep_learning._dummy_series_forecaster import (
    DummySeriesForecaster,
)
from aeon.forecasting.deep_learning.base import BaseDeepForecaster

Y_TEST = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])


def test_abstract_method_raises_error():
    """Checks that a forecaster must implement the abstract method."""

    class BrokenForecaster(BaseDeepForecaster, SeriesToSeriesForecastingMixin):
        def _fit(self, y, exog=None):
            return self

        def _predict(self, y, exog=None):
            return y[0, -1]

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BrokenForecaster(horizon=5, axis=1)


def test_series_to_series_forecast_output():
    """
    Test the functionality of the series_to_series_forecast method directly.

    This verifies that the mixin's final method works and the dummy forecaster
    correctly generates the full sequence in one go.
    """
    PH = 4
    EXPECTED_VALUE = 7.0
    f = DummySeriesForecaster(horizon=PH, axis=1, value_to_return=EXPECTED_VALUE)
    f.fit(Y_TEST)
    predictions = f.series_to_series_forecast(Y_TEST, PH)
    assert isinstance(predictions, np.ndarray) and predictions.shape == (PH,)
    expected_preds = np.full(PH, fill_value=EXPECTED_VALUE)
    np.testing.assert_array_equal(predictions, expected_preds)


def test_series_forecaster_base_predict():
    """Checks that the base class methods works correctly for single-step prediction."""
    f = DummySeriesForecaster(horizon=1, value_to_return=1.0)
    f.fit(Y_TEST)
    p1 = f.predict(Y_TEST)
    assert p1 == 1.0
    p2 = f.forecast(Y_TEST)
    assert p2 == 1.0
