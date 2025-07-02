"""Naive forecaster with parameters set on the training data."""

"""Naive forecaster with multiple strategies."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["AutoNaiveForecaster"]


import numpy as np

from aeon.forecasting.base import BaseForecaster
from aeon.forecasting._naive import NaiveForecaster

class AutoNaiveForecaster(BaseForecaster):
    """
    Naive forecaster with strategy set based on minimising error.

    Searches options, "last", "mean", and "seasonal_last", with season in range [2,max_season].
    If max_season is not passed to the constructor, it will be set to series length/2.

    Simple first implementation, splits the train series into 70% train and 30% validation split
    and minimises RMSE on the validation set.

    Parameters
    ----------
    max_season : int or None, default=None
        The maximum season to consider in the parameter search. In None, set as quarter the length of the series
        passed in `fit`.

    Examples
    --------
    >>> import aeon as ae
    """

    def __init__(self, max_season=None, horizon=1):
        self.max_season = max_season
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        y = y.squeeze()
        l = len(y)

        y_train = y[:int(0.7*l)]
        y_test = y[int(0.7*l):]
        # Eval last first
        last = y_train[-1]
        mean = np.mean(y_train)
        # measure error and pick one
        seasons = y_train[-self.max_season:] # Get all the fixed values

        return self

    def _predict(self, y=None, exog=None):
        if y is None:
            if self.strategy == "last" or self.strategy == "mean":
                return self._fitted_scalar_value

            # For "seasonal_last" strategy
            prediction_index = (self.horizon - 1) % self.seasonal_period
            return self._fitted_last_season[prediction_index]
        else:
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
