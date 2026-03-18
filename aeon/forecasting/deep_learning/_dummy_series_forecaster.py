"""Dummy forecaster for testing SeriesToSeriesForecastingMixin."""

import numpy as np

from aeon.forecasting.base import SeriesToSeriesForecastingMixin
from aeon.forecasting.deep_learning.base import BaseDeepForecaster


class DummySeriesForecaster(BaseDeepForecaster, SeriesToSeriesForecastingMixin):
    """
    A dummy forecaster used to test the SeriesToSeriesForecastingMixin.

    This forecaster always predicts a series of ones (or the last observed value)
    for the entire forecast horizon in a single step, demonstrating the mixin's
    functionality. The implementation ensures the new mixin's abstract method
    (_series_to_series_forecast) is implemented, proving the architecture
    is functional.
    """

    def __init__(
        self,
        horizon: int = 1,
        axis: int = 1,
        window: int = 5,
        value_to_return: float = 1.0,
    ):
        self.value_to_return = value_to_return
        super().__init__(horizon=horizon, window=window, axis=axis)

    def _fit(self, y, exog=None):
        return self

    def _predict(self, y, exog=None):
        """Just predict the simple constant value."""
        return self.value_to_return

    def _series_to_series_forecast(
        self, y, prediction_horizon, exog=None
    ) -> np.ndarray:
        """Return a 1D NumPy array with predicted value for entire horizon."""
        return np.full(prediction_horizon, fill_value=self.value_to_return)

    def build_model(self, input_shape):
        pass
