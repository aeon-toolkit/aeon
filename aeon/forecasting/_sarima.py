"""SARIMAForecaster.

An implementation of the Seasonal ARIMA forecasting algorithm.
"""

__maintainer__ = ["alexbanwell1", "TonyBagnall"]
__all__ = ["SARIMAForecaster"]

from math import comb

import numpy as np
from numba import njit

from aeon.forecasting import ARIMAForecaster
from aeon.forecasting._arima import _arima_model, _calc_arima, _extract_params
from aeon.utils.optimisation._nelder_mead import nelder_mead

NOGIL = False
CACHE = True


class SARIMAForecaster(ARIMAForecaster):
    """Seasonal AutoRegressive Integrated Moving Average (SARIMA) forecaster.

    Parameters
    ----------
    horizon : int, default=1
        The forecasting horizon, i.e., the number of steps ahead to predict.

    Attributes
    ----------
    ps_, ds_, qs_ : int
        Orders of the seasonal ARIMA model: seasonal AR (P), seasonal differencing (D),
        and seasonal MA (Q) terms.
    seasonal_period_ : int
        Length of the seasonal cycle.
    phi_s_ : array-like
        Coefficients for the seasonal autoregressive terms.
    theta_s_ : array-like
        Coefficients for the seasonal moving average terms.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
       Forecasting: Principles and Practice. OTexts, 2014.
       https://otexts.com/fpp3/

    Examples
    --------
    >>> from aeon.forecasting import SARIMAForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = SARIMAForecaster(1,1,2,0,1,0,12,0)
    >>> forecaster.fit(y)
    SARIMAForecaster(d=1, ds=1, q=2)
    >>> forecaster.predict()
    450.7487685084027
    """

    def __init__(
        self,
        p=1,
        d=0,
        q=1,
        ps=0,
        ds=0,
        qs=0,
        seasonal_period=12,
        constant_term=0,
        horizon=1,
    ):
        super().__init__(p=p, d=d, q=q, constant_term=constant_term, horizon=horizon)
        self.ps = ps
        self.ds = ds
        self.qs = qs
        self.seasonal_period = seasonal_period
        self.ps_ = 0
        self.ds_ = 0
        self.qs_ = 0
        self.seasonal_period_ = 0
        self.phi_s_ = 0
        self.theta_s_ = 0

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
        self.p_ = self.p
        self.d_ = self.d
        self.q_ = self.q
        self.ps_ = self.ps
        self.ds_ = self.ds
        self.qs_ = self.qs
        self.seasonal_period_ = self.seasonal_period
        if self.seasonal_period_ == 1:
            raise ValueError("Seasonal period must be greater than 1.")
        self.constant_term_ = self.constant_term
        self.data_ = np.array(y.squeeze(), dtype=np.float64)
        self.model_ = np.array(
            (
                self.constant_term,
                self.p,
                self.q,
                self.ps,
                self.qs,
                self.seasonal_period,
            ),
            dtype=np.int32,
        )
        self.differenced_data_ = np.diff(self.data_, n=self.d)
        for _ds in range(self.ds_):
            self.differenced_data_ = (
                self.differenced_data_[self.seasonal_period_ :]
                - self.differenced_data_[: -self.seasonal_period_]
            )
        (self.parameters_, self.aic_) = nelder_mead(
            _sarima_model_wrapper,
            np.sum(self.model_[:5]),
            self.differenced_data_,
            self.model_,
        )
        (self.c_, self.phi_, self.theta_, self.phi_s_, self.theta_s_) = _extract_params(
            self.parameters_, self.model_
        )
        (self.aic_, self.residuals_) = _arima_model(
            self.parameters_, _calc_sarima, self.differenced_data_, self.model_
        )
        return self

    def _predict(self, y=None, exog=None):
        """
        Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the next horizon value for. If None,
            predict the next horizon value after series seen in fit.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        y = np.array(y, dtype=np.float64)
        value = _calc_sarima(
            self.differenced_data_,
            self.model_,
            len(self.differenced_data_),
            _extract_params(self.parameters_, self.model_),
            self.residuals_,
        )
        history = self.data_[::-1]
        differenced_history = np.diff(self.data_, n=self.d_)[::-1]
        # Step 1: undo seasonal differencing on y^(d)
        for k in range(1, self.ds_ + 1):
            lag = k * self.seasonal_period_
            value += (-1) ** (k + 1) * comb(self.ds_, k) * differenced_history[lag - 1]

        # Step 2: undo ordinary differencing
        for k in range(1, self.d_ + 1):
            value += (-1) ** (k + 1) * comb(self.d_, k) * history[k - 1]
        return value.item()


@njit(fastmath=True)
def _sarima_model_wrapper(params, data, model):
    return _arima_model(params, _calc_sarima, data, model)[0]


@njit(cache=True, fastmath=True)
def _calc_sarima(data, model, t, formatted_params, residuals):
    """Calculate the SARIMA forecast for time t."""
    if len(model) != 6:
        raise ValueError("Model must be of the form (c, p, q, ps, qs, seasonal_period)")
    arima_forecast = _calc_arima(data, model[:3], t, formatted_params, residuals)
    seasonal_period = model[5]
    # Seasonal AR part
    ps = model[3]
    phi_s = formatted_params[3][:ps]
    ars_term = (
        0
        if (t - seasonal_period * ps) < 0
        else np.dot(phi_s, data[t - seasonal_period * ps : t : seasonal_period][::-1])
    )
    # Seasonal MA part
    qs = model[4]
    theta_s = formatted_params[4][:qs]
    mas_term = (
        0
        if (t - seasonal_period * qs) < 0
        else np.dot(
            theta_s, residuals[t - seasonal_period * qs : t : seasonal_period][::-1]
        )
    )
    return arima_forecast + ars_term + mas_term
