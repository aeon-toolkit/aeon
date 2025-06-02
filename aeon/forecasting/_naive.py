"""Naive Forecaster."""

import numpy as np

from aeon.forecasting.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """
    Naive forecaster with multiple strategies and flexible horizon.

    Strategies:
    - "last": predicts the last seen value in training for all horizon steps.

    - "mean": predicts the mean of the training series for all horizon steps.

    - "seasonal_last": predict the last season value seen in the training series.
                       Returns np.nan if the effective seasonal data is empty.
    """

    def __init__(self, strategy="last", seasonal_period=1, horizon=1):
        """
        Initialize NaiveForecaster.

        Parameters
        ----------
        strategy : str, default="last"
            The forecasting strategy to use.
            Options: "last", "mean", "seasonal_last".
        seasonal_period : int, default=1
            The seasonal period to use for the "seasonal_last" strategy.
            E.g., 12 for monthly data with annual seasonality.
        horizon : int, default=1
            The number of time steps ahead to forecast.
        """
        self.strategy = strategy
        self.seasonal_period = seasonal_period
        self._fitted_scalar_value_ = None  # For 'last' and 'mean' strategies
        self._fitted_last_season_ = np.array([])  # For 'seasonal_last', init as empty

        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        """Fit Naive forecaster to training data `y`."""
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
        """Predict with the fitted Naive forecaster."""
        predictions = np.zeros(self.horizon)

        if self.strategy == "last" or self.strategy == "mean":
            predictions[:] = self._fitted_scalar_value_
        elif self.strategy == "seasonal_last":
            m = len(self._fitted_last_season_)
            for i in range(self.horizon):
                predictions[i] = self._fitted_last_season_[i % m]
        return predictions

    def _forecast(self, y, exog=None):
        """Forecast using Naive forecaster on new data `y` for `self.horizon` steps."""
        y_squeezed = y.squeeze()
        predictions = np.zeros(self.horizon)

        if self.strategy == "last":
            predictions[:] = y_squeezed[-1]
        elif self.strategy == "mean":
            predictions[:] = np.mean(y_squeezed)
        elif self.strategy == "seasonal_last":
            last_season_from_input = y_squeezed[-self.seasonal_period :]
            m = len(last_season_from_input)
            for i in range(self.horizon):
                predictions[i] = last_season_from_input[i % m]
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                "Valid strategies are 'last', 'mean', 'seasonal_last'."
            )
        return predictions
