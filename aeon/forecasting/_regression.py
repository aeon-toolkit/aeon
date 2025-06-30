"""Window-based regression forecaster.

General purpose forecaster to use with any scikit learn or aeon compatible
regressor. Simply forms a collection of series using windowing from the time series
to form ``X`` and trains to predict the next ``horizon`` points ahead.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.forecasting.base import (
    BaseForecaster,
    DirectForecastingMixin,
    IterativeForecastingMixin,
)


class RegressionForecaster(
    BaseForecaster, DirectForecastingMixin, IterativeForecastingMixin
):
    """
    Regression based forecasting.

    Container for forecaster that reduces forecasting to regression through a
    window. Form a collection of sub-series of length ``window`` through a sliding
    window to form training collection ``X``, take ``horizon`` points ahead to form
    ``y``, then apply an aeon or sklearn regressor.

    If exogenous variables are provided, they are concatenated with the main series
    and included in the regression windows.

    Parameters
    ----------
    window : int
        The window prior to the current time point to use in forecasting. So if
        horizon is one, forecaster will train using points $i$ to $window+i-1$ to
        predict value $window+i$. If horizon is 4, forecaster will used points $i$
        to $window+i-1$ to predict value $window+i+3$.
    horizon : int, default =1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead
    regressor : object, default =None
        Regression estimator that implements BaseRegressor or is otherwise compatible
        with sklearn regressors.
    """

    _tags = {
        "capability:exogenous": True,
    }

    def __init__(self, window: int, horizon: int = 1, regressor=None):
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
            Optional exogenous time series data, assumed to be aligned with y.

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

        # Combine y and exog for windowing
        if exog is not None:
            if exog.ndim == 1:
                exog = exog.reshape(1, -1)
            if exog.shape[1] != y.shape[1]:
                raise ValueError("y and exog must have the same number of time points.")
            combined_data = np.vstack([y, exog])
        else:
            combined_data = y

        # Enforce a minimum number of training samples, currently 3
        if self.window < 1 or self.window >= combined_data.shape[1] - 3:
            raise ValueError(
                f"window value {self.window} is invalid for series length "
                f"{combined_data.shape[1]}"
            )

        # Create windowed data for X
        X = np.lib.stride_tricks.sliding_window_view(
            combined_data, window_shape=(combined_data.shape[0], self.window)
        )
        X = X.squeeze(axis=0)
        X = X[:, :, :].reshape(X.shape[0], -1)

        # Ignore the final horizon values for X
        X = X[: -self.horizon]

        # Extract y_train from the original series
        y_train = y.squeeze()[self.window + self.horizon - 1 :]

        self.last_ = combined_data[:, -self.window :]
        self.regressor_.fit(X=X, y=y_train)
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
            Optional exogenous time series data, assumed to be aligned with y.

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        if y is None:
            # Flatten the last window to be compatible with sklearn regressors
            last_window_flat = self.last_.reshape(1, -1)
            return self.regressor_.predict(last_window_flat)[0]

        # Combine y and exog for prediction
        if exog is not None:
            if exog.ndim == 1:
                exog = exog.reshape(1, -1)
            if exog.shape[1] != y.shape[1]:
                raise ValueError("y and exog must have the same number of time points.")
            combined_data = np.vstack([y, exog])
        else:
            combined_data = y

        # Extract the last window and flatten for prediction
        last_window = combined_data[:, -self.window :]
        last_window_flat = last_window.reshape(1, -1)

        return self.regressor_.predict(last_window_flat)[0]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
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
