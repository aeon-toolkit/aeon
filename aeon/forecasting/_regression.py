"""Window-based regression forecaster.

General purpose forecaster to use with any scikit learn or aeon compatible
regressor. Simply forms a collection of windows from the time series and trains to
predict the next
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import BaseForecaster


class RegressionForecaster(BaseForecaster):
    """
    Regression based forecasting.

    Container for forecaster that reduces forecasting to regression through a
    window. Form a collection of sub series of length `window` through a sliding
    winodw to form X, take `horizon` points ahead to form `y`, then apply an aeon or
    sklearn regressor.


    Parameters
    ----------
    window : int
        The window prior to the current time point to use in forecasting. So if
        horizon is one, forecaster will train using points $i$ to $window+i-1$ to
        predict value $window+i$. If horizon is 4, forecaster will used points $i$
        to $window+i-1$ to predict value $window+i+3$. If None, the algorithm will
        internally determine what data to use to predict `horizon` steps ahead.
    horizon : int, default =1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead
    regressor : object, default =None
        Regression estimator that implements BaseRegressor or is otherwise compatible
        with sklearn regressors.
    """

    def __init__(self, window, horizon=1, regressor=None):
        self.window = window
        self.regressor = regressor
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        """Fit forecaster to time series.

        Split X into windows of length window and train the forecaster on each window
        to predict the horizon ahead.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead.
        exog : np.ndarray, default=None
            Optional exogenous time series data. Included for interface
            compatibility but ignored in this estimator.

        Returns
        -------
        self
            Fitted estimator
        """
        # Window data
        if self.regressor is None:
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = self.regressor
        y = y.squeeze()
        X = np.lib.stride_tricks.sliding_window_view(y, window_shape=self.window)
        # Ignore the final horizon values: need to store these for pred with empty y
        X = X[: -self.horizon]
        # Extract y
        y = y[self.window + self.horizon - 1 :]
        self.last_ = y[-self.window :]
        self.last_ = self.last_.reshape(1, -1)
        self.regressor_.fit(X=X, y=y)
        return self

    def _predict(self, y=None, exog=None):
        """
        Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the next horizon value for. If None,
            predict the next horizon value after series seen in fit.
        exog : np.ndarray, default=None
            Optional exogenous time series data. Included for interface
            compatibility but ignored in this estimator.

        Returns
        -------
        np.ndarray
            single prediction self.horizon steps ahead of y.
        """
        if y is None:
            return self.regressor_.predict(self.last_)
        last = y[:, -self.window :]
        return self.regressor_.predict(last)

    def _forecast(self, y, exog=None):
        """
        Forecast the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray
            A time series to predict the next horizon value for.
        exog : np.ndarray, default=None
            Optional exogenous time series data. Included for interface
            compatibility but ignored in this estimator.

        Returns
        -------
        np.ndarray
            single prediction self.horizon steps ahead of y.

        NOTE: deal with horizons
        """
        self.fit(y, exog)
        return self.predict()

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default='default'
            Name of the parameter set to return.

        Returns
        -------
        dict
            Dictionary of testing parameter settings.
        """
        return {"window": 4}
