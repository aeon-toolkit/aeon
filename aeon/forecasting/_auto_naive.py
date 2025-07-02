"""Naive forecaster with parameters set on the training data."""

"""Naive forecaster with multiple strategies."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["AutoNaiveForecaster"]


import numpy as np
from enum import Enum
from aeon.forecasting.base import BaseForecaster


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
        self.strategy_ = "last"
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        y = y.squeeze()
        # last strategy
        mse_last = np.mean((y[1:] - y[:-1]) ** 2)

        # series mean strategy, in sample
        mse_mean = np.mean((y - np.mean(y)) ** 2)

        # seasonal strategy, in sample
        max_season = self.max_season
        if self.max_season is None:
            max_season = len(y)/4
        best_s = None
        best_seasonal = np.inf
        for s in range(1, max_season + 1):
            # Predict y[t] = y[t - s]
            y_true = y[s:]
            y_pred = y[:-s]
            mse = np.mean((y_true - y_pred) ** 2)

            if mse < best_seasonal:
                best_seasonal = mse
                best_s = s
        self.best_mse_ = mse_last
        self._fitted_scalar_value = y[:-1]

        if mse_mean < mse_last:
            self.strategy_ = "mean"
            self.best_mse_ = mse_mean
            self._fitted_scalar_value = np.mean(y)

        if self.best_mse_ < best_seasonal:
            self.strategy_ = "seasonal"
            self.season = best_s
            self.best_mse_ = best_seasonal

        return self

    def _predict(self, y, exog=None):
        if self.strategy_ == "last" or self.strategy_ == "mean":
                    return self._fitted_scalar_value
            # For "seasonal_last" strategy
        prediction_index = (self.horizon - 1) % self.seasonal_period
        return self._fitted_last_season[prediction_index]
