"""Window-based regression forecaster.

General purpose forecaster to use with any scikit learn or aeon compatible
regressor. Simply forms a collection of series using windowing from the time series
to form ``X`` and trains to predict the next ``horizon`` points ahead.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.base._base import _clone_estimator
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

    If exogenous variables are provided, they are used as target-time features
    aligned with the prediction target. Historical exogenous effects should be
    represented by explicitly lagging exogenous variables before passing them to the
    forecaster; exogenous variables are not treated as extra lag-window channels.

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
            self.regressor_ = _clone_estimator(self.regressor)
        self._n_exog = 0
        y_1d = y.squeeze()

        n_timepoints = y_1d.shape[0]
        exog_target = None
        if exog is not None:
            exog_target = self._format_fit_exog(exog, n_timepoints)
            self._n_exog = exog_target.shape[1]

        # Enforce a minimum number of training samples, currently 3
        if self.window < 1 or self.window >= n_timepoints - self.horizon - 2:
            raise ValueError(
                f"window value {self.window} is invalid for series length "
                f"{n_timepoints}"
            )

        # Create lagged y windows and append target-time exogenous features.
        X_train = np.lib.stride_tricks.sliding_window_view(
            y_1d, window_shape=self.window
        )[: -self.horizon]
        target_indices = np.arange(self.window + self.horizon - 1, n_timepoints)
        if exog_target is not None:
            X_train = np.hstack([X_train, exog_target[target_indices]])

        # Extract y_train from the original series
        y_train = y_1d[target_indices]

        self.regressor_.fit(X=X_train, y=y_train)

        last_y_window = y_1d[-self.window :].reshape(1, -1)
        if exog_target is not None:
            last_exog = exog_target[[-1]]
            last_y_window = np.hstack([last_y_window, last_exog])
        self.forecast_ = self.regressor_.predict(last_y_window)[0]
        return self

    def _predict(self, y, exog=None):
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
        y = y[:, -self.window :]
        y = y.squeeze()
        # Test data compliant for regression based
        if len(y) < self.window:
            raise ValueError(
                f" Series passed in predict length = {len(y)} but this "
                f"RegressionForecaster was trained on window length = "
                f"{self.window}"
            )
        features = y.reshape(1, -1)
        if exog is not None:
            if self._n_exog == 0:
                raise ValueError(
                    "predict passed exogenous variables, but this "
                    "RegressionForecaster was fitted without exog"
                )
            exog_row = self._format_predict_exog(exog)
            features = np.hstack([features, exog_row])
        else:
            if self._n_exog > 0:
                raise ValueError(
                    f" predict passed no exogenous variables, but this "
                    f"RegressionForecaster was trained on {self._n_exog} exog in fit"
                )

        # Extract the last window and flatten for prediction
        last_window = features.reshape(1, -1)

        return self.regressor_.predict(last_window)[0]

    def _forecast(self, y, exog=None):
        """Forecast values for time series X."""
        self.fit(y, exog)
        return self.forecast_

    @staticmethod
    def _format_fit_exog(exog, n_timepoints):
        """Convert fit exog to timepoint rows."""
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] == n_timepoints:
            return exog
        if exog.shape[0] == 1 and exog.shape[1] == n_timepoints:
            return exog.T
        raise ValueError(
            "exog must contain one row per time point in y. "
            f"Got {exog.shape[0]}, expected {n_timepoints}."
        )

    def _format_predict_exog(self, exog):
        """Convert prediction exog to a single target-time row."""
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != 1:
            raise ValueError("exog for predict must contain a single target-time row.")
        if exog.shape[1] != self._n_exog:
            raise ValueError(
                "exog for predict must contain a single target-time row "
                f"with {self._n_exog} features, got {exog.shape[1]}."
            )
        return exog

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
