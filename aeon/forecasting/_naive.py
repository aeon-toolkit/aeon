"""Naive Forecaster."""

import numpy as np

from aeon.forecasting.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """
    Naive forecaster with multiple strategies and flexible horizon.

    Parameters
    ----------
    strategy : str, default="last"
        The forecasting strategy to use.
        Options: "last", "mean", "seasonal_last".
            - "last" predicts the last seen value in training for all horizon steps.
            - "mean": predicts the mean of the training series for all horizon steps.
            - "seasonal_last": predicts the last season value in the training series.
              Returns np.nan if the effective seasonal data is empty.
    seasonal_period : int, default=1
        The seasonal period to use for the "seasonal_last" strategy.
        E.g., 12 for monthly data with annual seasonality.
    horizon : int, default=1
        The number of time steps ahead to forecast.
    """

    def __init__(self, strategy="last", seasonal_period=1, horizon=1):
        """Initialize NaiveForecaster."""
        self.strategy = strategy
        self.seasonal_period = seasonal_period

        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        """Fit Naive forecaster to training data `y`."""
        self._fitted_scalar_value_ = None  # For 'last' and 'mean' strategies
        self._fitted_last_season_ = np.array([])  # For 'seasonal_last', init as empty

        y_squeezed = y.squeeze()

        if self.strategy == "last":
            self._fitted_scalar_value_ = y_squeezed[-1]
        elif self.strategy == "mean":
            self._fitted_scalar_value_ = np.mean(y_squeezed)
        elif self.strategy == "seasonal_last":
            self._fitted_last_season_ = y_squeezed[-self.seasonal_period :]
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                "Valid strategies are 'last', 'mean', 'seasonal_last'."
            )
        return self

    def _predict(self, y=None, exog=None):
        """Predict a single value self.horizon steps ahead."""
        if self.strategy == "last" or self.strategy == "mean":
            return self._fitted_scalar_value_

        # For "seasonal_last" strategy
        prediction_index = (self.horizon - 1) % self.seasonal_period
        return self._fitted_last_season_[prediction_index]

    # The class will inherit the _forecast method from BaseForecaster.
