"""Window-based regression forecaster on ES normalised data.

Wraps any scikit-learn or aeon compatible regressor. Windows of the series are
normalised by the level and seasonal factors of an exponential smoothing model
before being passed to the regressor, and its prediction is mapped back to the
original scale, following the preprocessing used in ES-RNN [1]_.
"""

__maintainer__ = ["AlexBanwell"]
__all__ = ["ESNormalisedRegressionForecaster"]

import numpy as np
from sklearn.linear_model import LinearRegression

from aeon.base._base import _clone_estimator
from aeon.forecasting.base import (
    BaseForecaster,
    DirectForecastingMixin,
    IterativeForecastingMixin,
)
from aeon.forecasting.utils._es_normalise import (
    es_components,
    es_denormalise,
    es_normalise_windows,
    fit_es_components,
)


class ESNormalisedRegressionForecaster(
    BaseForecaster, DirectForecastingMixin, IterativeForecastingMixin
):
    """Regression forecaster on exponential smoothing normalised windows.

    An ETS model with multiplicative error, no trend and multiplicative seasonality
    is fitted to the series to estimate its level and seasonal factors. Each sliding
    window ending at time ``t`` is divided by the level at ``t`` and by each point's
    seasonal factor, and the target ``horizon`` steps ahead is divided by the same
    level and its seasonal factor. The regressor is trained on these scale-free
    windows, and its forecast is multiplied back by the final level and the
    seasonal factor of the forecast time step. ES has no trend component, so any
    trend is left for the regressor to learn.

    Parameters
    ----------
    window : int
        Number of past points used as features. If the series is too short to give
        at least three training windows for the current ``horizon``, the window is
        reduced to fit; the value used is stored in ``window_``.
    horizon : int, default=1
        Number of steps ahead to forecast.
    regressor : object, default=None
        Regression estimator that implements BaseRegressor or is otherwise compatible
        with sklearn regressors. If None, ``LinearRegression`` is used.
    seasonal_period : int, default=1
        Number of time points in a seasonal cycle. If 1, or if the series is shorter
        than two full seasons, only the level is used for normalisation.
    alpha : float or None, default=None
        Level smoothing parameter. If None it is estimated by fitting the ETS model.
    gamma : float or None, default=None
        Seasonal smoothing parameter. If None it is estimated by fitting the ETS
        model.
    log : bool, default=False
        If True, the regressor is fitted on the log of the normalised values.

    Attributes
    ----------
    regressor_ : object
        Fitted regressor.
    alpha_ : float
        Level smoothing parameter used.
    gamma_ : float
        Seasonal smoothing parameter used.
    seasonal_period_ : int
        Seasonal period used, 1 if seasonality was dropped.
    window_ : int
        Window length used.
    forecast_ : float
        Forecast ``horizon`` steps ahead of the series passed to ``fit``.

    Notes
    -----
    The series must be strictly positive. ES-RNN learns the smoothing parameters
    jointly with the network; here they are fitted beforehand by maximum likelihood.

    References
    ----------
    .. [1] Smyl, S. (2020). A hybrid method of exponential smoothing and recurrent
       neural networks for time series forecasting. International Journal of
       Forecasting, 36(1), 75-85.

    Examples
    --------
    >>> from aeon.forecasting import ESNormalisedRegressionForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> f = ESNormalisedRegressionForecaster(window=24, seasonal_period=12)
    >>> preds = f.direct_forecast(y, 3)
    """

    _tags = {
        "capability:exogenous": False,
    }

    def __init__(
        self,
        window: int,
        horizon: int = 1,
        regressor=None,
        seasonal_period: int = 1,
        alpha: float | None = None,
        gamma: float | None = None,
        log: bool = False,
    ):
        self.window = window
        self.regressor = regressor
        self.seasonal_period = seasonal_period
        self.alpha = alpha
        self.gamma = gamma
        self.log = log
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        """Fit the ES model and the regressor on normalised windows."""
        y = y.squeeze()
        n_timepoints = y.shape[0]
        if self.window < 1:
            raise ValueError(f"window must be at least 1, got {self.window}.")
        # Enforce a minimum of three training windows, as RegressionForecaster does
        max_window = n_timepoints - self.horizon - 2
        if max_window < 1:
            raise ValueError(
                f"Series of length {n_timepoints} is too short for horizon "
                f"{self.horizon}."
            )
        self.window_ = min(self.window, max_window)

        level, season, self.alpha_, self.gamma_, self.seasonal_period_ = (
            fit_es_components(y, self.seasonal_period, self.alpha, self.gamma)
        )
        X, z = es_normalise_windows(
            y,
            level,
            season,
            self.window_,
            self.horizon,
            self.seasonal_period_,
            log=self.log,
        )
        if self.regressor is None:
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = _clone_estimator(self.regressor)
        self.regressor_.fit(X=X, y=z)
        self.forecast_ = self._predict_from_states(y, level, season)
        return self

    def _predict(self, y, exog=None):
        """Predict ``horizon`` steps ahead of ``y`` using the fitted ES parameters."""
        y = y.squeeze()
        if len(y) < self.window_:
            raise ValueError(
                f"Series passed in predict length = {len(y)} but this "
                f"forecaster was trained on window length = {self.window_}"
            )
        level, season = es_components(
            y, self.seasonal_period_, self.alpha_, self.gamma_
        )
        if len(season) - len(y) != self.seasonal_period_:
            raise ValueError(
                f"Series passed in predict length = {len(y)} is too short for the "
                f"seasonal period {self.seasonal_period_} used in fit."
            )
        return self._predict_from_states(y, level, season)

    def _predict_from_states(self, y, level, season):
        origin = len(y) - 1
        start = origin - self.window_ + 1
        x = y[start:] / (level[origin] * season[start : origin + 1])
        if self.log:
            x = np.log(x)
        z = self.regressor_.predict(x.reshape(1, -1))[0]
        return float(
            es_denormalise(
                z,
                level,
                season,
                origin,
                self.horizon,
                self.seasonal_period_,
                log=self.log,
            )
        )

    def _forecast(self, y, exog=None):
        """Forecast ``horizon`` steps ahead of ``y``."""
        self.fit(y, exog)
        return self.forecast_

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
        return {"window": 4, "seasonal_period": 4}
