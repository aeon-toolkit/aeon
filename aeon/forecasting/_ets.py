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
    season_len : int, default = 1
        The length of the seasonality period.
    """

    def __init__(self, alpha=0.2, beta=0.2, gamma=0.2, season_len=1, horizon=1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_len = season_len
        self.forecast_val_ = 0.0
        self.level_ = 0.0
        self.trend_ = 0.0
        self.season_ = None
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
        sl = self.season_len
        # Initialize components
        self.level_ = data[0]
        self.trend_ = np.mean(data[sl : 2 * sl]) - np.mean(data[:sl])
        self.season_ = [data[i] / data[0] for i in range(sl)]
        for t in range(sl, self.n_timepoints):
            # Calculate level, trend, and seasonal components
            level_prev = self.level_
            l1 = data[t] / self.season_[t % self.season_len]
            l2 = self.level_ + self.trend_
            self.level_ = self.alpha * l1 + (1 - self.alpha) * l2
            trend = self.level_ - level_prev
            self.trend_ = self.beta * trend + (1 - self.beta) * self.trend_
            s1 = data[t] / self.level_
            s2 = self.season_[t % sl]
            self.season_[t % self.season_len] = self.gamma * s1 + (1 - self.gamma) * s2
        return self

    def _predict(self, y=None, exog=None):
        # Generate forecasts based on the final values of level, trend, and seasonals
        trend = (self.horizon + 1) * self.trend_
        seasonal = self.season_[(self.n_timepoints + self.horizon) % self.season_len]
        forecast = (self.level_ + trend) * seasonal
        return forecast

    def _forecast(self, y):
        self.fit(y)
        return self.predict()
