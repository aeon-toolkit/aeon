"""Naive forecaster with multiple strategies."""

__maintainer__ = []
__all__ = ["NaiveForecaster"]


import numpy as np

from aeon.forecasting.base import (
    BaseForecaster,
    DirectForecastingMixin,
    IterativeForecastingMixin,
)


class NaiveForecaster(
    BaseForecaster, DirectForecastingMixin, IterativeForecastingMixin
):
    """
    Naive forecaster with multiple strategies and flexible horizon.

    Parameters
    ----------
    strategy : str, default="last"
        The forecasting strategy to use.
        Options: "last", "mean", "seasonal_last".
            - "last" predicts the last value of the input series for all horizon steps.
            - "mean": predicts the mean of the input series for all horizon steps.
            - "seasonal_last": predicts the last season value in the training series.
              Returns np.nan if the effective seasonal data is empty.
    seasonal_period : int, default=1
        The seasonal period to use for the "seasonal_last" strategy.
        E.g., 12 for monthly data with annual seasonality.
    horizon : int, default =1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead.
        Only relevant for "seasonal_last".
    """

    _tags = {
        "fit_is_empty": True,
    }

    def __init__(self, strategy="last", seasonal_period=1, horizon=1):
        self.strategy = strategy
        self.seasonal_period = seasonal_period

        super().__init__(horizon=horizon, axis=1)

    def _predict(self, y, exog=None):
        y_squeezed = y.squeeze()

        if self.strategy == "last":
            return y_squeezed[-1]
        elif self.strategy == "mean":
            return np.mean(y_squeezed)
        elif self.strategy == "seasonal_last":
            period = y_squeezed[-self.seasonal_period :]
            idx = (self.horizon - 1) % self.seasonal_period
            return period[idx]
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                "Valid strategies are 'last', 'mean', 'seasonal_last'."
            )
