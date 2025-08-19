"""AutoARIMA.

An implementation of the AutoARIMA forecasting algorithm.
"""

__maintainer__ = ["alexbanwell1", "TonyBagnall"]
__all__ = ["AutoARIMA"]

import numpy as np
from numba import njit

from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.stats import ARIMA
from aeon.forecasting.utils._hypo_tests import kpss_test
from aeon.forecasting.utils._nelder_mead import nelder_mead


class AutoARIMA(BaseForecaster):
    """AutoRegressive Integrated Moving Average (ARIMA) forecaster.

    Implements the Hyndman-Khandakar automatic ARIMA algorithm for time series
    forecasting with optional seasonal components. The model automatically selects
    the orders of the (p, d, q) components based on information criteria, such as AIC.

    Parameters
    ----------
    horizon : int, default=1
        The forecasting horizon, i.e., the number of steps ahead to predict.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
       Forecasting: Principles and Practice. OTexts, 2014.
       https://otexts.com/fpp3/

    Examples
    --------
    >>> from aeon.forecasting.stats import AutoARIMA
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AutoARIMA()
    >>> forecaster.forecast(y)
    482.1873214419662
    """

    def __init__(self):
        self.p_ = 0
        self.d_ = 0
        self.q_ = 0
        self.constant_term_ = False
        self.wrapped_model_ = None
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit AutoARIMA forecaster to series y.

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
            Fitted ARIMAForecaster.
        """
        series = np.array(y.squeeze(), dtype=np.float64)
        differenced_series = series.copy()
        self.d_ = 0
        while not kpss_test(differenced_series)[1]:
            differenced_series = np.diff(differenced_series, n=1)
            self.d_ += 1
        include_constant = 1 if self.d_ == 0 else 0
        model_parameters = np.array(
            [
                [include_constant, 2, 2],
                [include_constant, 0, 0],
                [include_constant, 1, 0],
                [include_constant, 0, 1],
            ]
        )
        (
            model,
            _,
            _,
        ) = _auto_arima(differenced_series, 0, model_parameters, 3)
        (
            constant_term_int,
            self.p_,
            self.q_,
        ) = model
        self.constant_term_ = constant_term_int == 1
        self.wrapped_model_ = ARIMA(self.p_, self.d_, self.q_, self.constant_term_)
        self.wrapped_model_.fit(y)
        return self

    def _predict(self, y, exog=None):
        """
        Predict the next step ahead for y.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the value of. y can be independent of the series
            seen in fit.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            Prediction 1 step ahead of the last value in y.
        """
        return self.wrapped_model_.predict(y, exog)

    def _forecast(self, y, exog=None):
        """Forecast one ahead for time series y."""
        self.fit(y, exog)
        return float(self.wrapped_model_.forecast_)

    def iterative_forecast(self, y, prediction_horizon):
        """Forecast ``prediction_horizon`` prediction using a single model fit on `y`.

        This function implements the iterative forecasting strategy (also called
        recursive or iterated). This involves a single model fit on ``y`` which is then
        used to make ``prediction_horizon`` ahead forecasts using its own predictions as
        inputs for future forecasts. This is done by taking the prediction at step
        ``i`` and feeding it back into the model to help predict for step ``i+1``.
        The basic contract of `iterative_forecast` is that `fit` is only ever called
        once.

        y : np.ndarray
            The time series to make forecasts about.  Must be of shape
            ``(n_channels, n_timepoints)`` if a multivariate time series.
        prediction_horizon : int
            The number of future time steps to forecast.

        Returns
        -------
        np.ndarray
            An array of shape `(prediction_horizon,)` containing the forecasts for
            each horizon.

        Raises
        ------
        ValueError
            if prediction_horizon` less than 1.
        """
        return self.wrapped_model_.iterative_forecast(y, prediction_horizon)


@njit(cache=True, fastmath=True)
def _auto_arima(
    differenced_data, loss_function, inital_model_parameters, num_model_params=3
):
    """
    Implement the Hyndman-Khandakar algorithm.

    For automatic ARIMA model selection.
    """
    best_score = -1
    best_model = inital_model_parameters[0]
    best_points = None
    for model in inital_model_parameters:
        points, aic = nelder_mead(
            loss_function,
            np.sum(model[:num_model_params]),
            differenced_data,
            model,
        )
        if (aic < best_score) or (best_score == -1):
            best_score = aic
            best_model = model
            best_points = points

    while True:
        better_model = False
        for param_no in range(1, num_model_params):
            for adjustment in [-1, 1]:
                if (best_model[param_no] + adjustment) < 0:
                    continue
                model = best_model.copy()
                model[param_no] += adjustment
                for constant_term in [0, 1]:
                    model[0] = constant_term
                    points, aic = nelder_mead(
                        loss_function,
                        np.sum(model[:num_model_params]),
                        differenced_data,
                        model,
                    )
                    if aic < best_score:
                        best_model = model.copy()
                        best_points = points
                        best_score = aic
                        better_model = True
        if not better_model:
            break
    return (
        best_model,
        best_points,
        best_score,
    )
