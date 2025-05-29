"""AutoSARIMAForecaster.

An implementation of the Auto SARIMA forecasting algorithm.
"""

__maintainer__ = ["alexbanwell1", "TonyBagnall"]
__all__ = ["AutoSARIMAForecaster"]

import numpy as np

from aeon.forecasting._arima import _arima_model, _extract_params
from aeon.forecasting._auto_arima import _auto_arima
from aeon.forecasting._sarima import (
    SARIMAForecaster,
    _calc_sarima,
    _sarima_model_wrapper,
)
from aeon.utils.forecasting._hypo_tests import kpss_test
from aeon.utils.forecasting._seasonality import calc_seasonal_period

NOGIL = False
CACHE = True


class AutoSARIMAForecaster(SARIMAForecaster):
    """Seasonal AutoRegressive Integrated Moving Average (SARIMA) forecaster.

    Implements the Hyndman-Khandakar automatic ARIMA algorithm for time series
    forecasting with optional seasonal components. The model automatically selects
    the orders of the non-seasonal (p, d, q) and seasonal (P, D, Q, m) components
    based on information criteria, such as AIC.

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
    >>> from aeon.forecasting import AutoSARIMAForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AutoSARIMAForecaster()
    >>> forecaster.fit(y)
    AutoSARIMAForecaster()
    >>> forecaster.predict()
    450.74890...
    """

    def __init__(self, horizon=1):
        super().__init__(horizon=horizon)

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
        self.data_ = np.array(y.squeeze(), dtype=np.float64)
        self.seasonal_period_ = calc_seasonal_period(self.data_)
        self.differenced_data_ = self.data_.copy()
        self.d_ = 0
        while not kpss_test(self.differenced_data_)[1]:
            self.differenced_data_ = np.diff(self.differenced_data_, n=1)
            self.d_ += 1
        self.ds_ = 1 if self.seasonal_period_ > 1 else 0
        if self.ds_:
            self.differenced_data_ = (
                self.differenced_data_[self.seasonal_period_ :]
                - self.differenced_data_[: -self.seasonal_period_]
            )
        include_c = 1 if self.d_ == 0 else 0
        model_parameters = np.array(
            [
                [include_c, 2, 2, 0, 0, self.seasonal_period_],
                [include_c, 0, 0, 0, 0, self.seasonal_period_],
                [include_c, 1, 0, 0, 0, self.seasonal_period_],
                [include_c, 0, 1, 0, 0, self.seasonal_period_],
            ]
        )
        (
            self.model_,
            self.parameters_,
            self.aic_,
        ) = _auto_arima(
            self.differenced_data_, _sarima_model_wrapper, model_parameters, 5
        )
        (
            self.constant_term_,
            self.p_,
            self.q_,
            self.ps_,
            self.qs_,
            self.seasonal_period_,
        ) = self.model_
        (self.c_, self.phi_, self.theta_, self.phi_s_, self.theta_s_) = _extract_params(
            self.parameters_, self.model_
        )
        (
            self.aic_,
            self.residuals_,
        ) = _arima_model(
            self.parameters_, _calc_sarima, self.differenced_data_, self.model_
        )
        return self
