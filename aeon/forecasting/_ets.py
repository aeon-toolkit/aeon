import numpy as np

from aeon.forecasting.base import BaseForecaster


class ETSForecaster(BaseForecaster):
    """Exponential Smoothing forecaster.

    Simple first implementation with Holt-Winters method
    and no seasonality.

    Parameters
    ----------
    alpha : float, default = 0.2
        Level smoothing parameter.
    beta : float, default = 0.2
        Trend smoothing parameter.
    gamma : float, default = 0.2
        Seasonal smoothing parameter.
    season_length : int, default = 1
        The length of the seasonality period.
    """

    def __init__(self, alpha=0.2, beta=0.2, gamma=0.2, season_length=1, horizon=1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.forecast_val_ = 0.0
        self.level_ = 0.0
        self.trend_ = 0.0
        self.seasonals_ = None
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        """Fit Exponential Smoothing forecaster to series y.

        Fit a forecaster to predict self.horizon steps ahead using y.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        self
            Fitted BaseForecaster.
        """
        data = y.squeeze()
        self.n_timepoints = len(data)
        sl = self.season_length
        # Initialize components
        self.level_ = data[0]
        self.trend_ = np.mean(data[sl : 2 * sl]) - np.mean(data[:sl])
        self.seasonals_ = [data[i] / data[0] for i in range(sl)]
        for t in range(sl, self.n_timepoints):
            # Calculate level, trend, and seasonal components
            level_prev = self.level_
            self.level_ = self.alpha * (
                data[t] / self.seasonals_[t % self.season_length]
            ) + (1 - self.alpha) * (self.level_ + self.trend_)
            self.trend_ = (
                self.beta * (self.level_ - level_prev) + (1 - self.beta) * self.trend_
            )
            self.seasonals_[t % self.season_length] = (
                self.gamma * (data[t] / self.level_)
                + (1 - self.gamma) * self.seasonals_[t % sl]
            )
        return self

    def _predict(self, y=None, exog=None):
        # Generate forecasts based on the final values of level, trend, and seasonals
        forecast = (self.level_ + (self.horizon + 1) * self.trend_) * self.seasonals_[
            (self.n_timepoints + self.horizon) % self.season_length
        ]
        return forecast

    def _forecast(self, y):
        self.fit(y)
        return self.predict()
