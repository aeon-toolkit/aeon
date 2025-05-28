"""AutoARIMAForecaster.

An implementation of the AutoARIMA forecasting algorithm.
"""

__maintainer__ = ["alexbanwell1", "TonyBagnall"]
__all__ = ["AutoARIMAForecaster"]

import numpy as np

from aeon.forecasting import ARIMAForecaster
from aeon.forecasting._arima import (
    _arima_model,
    _arima_model_wrapper,
    _calc_arima,
    _extract_params,
)
from aeon.utils.forecasting._hypo_tests import kpss_test
from aeon.utils.optimisation._nelder_mead import nelder_mead


class AutoARIMAForecaster(ARIMAForecaster):
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
    >>> from aeon.forecasting import AutoARIMAForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AutoARIMAForecaster()
    >>> forecaster.fit(y)
    AutoARIMAForecaster()
    >>> forecaster.predict()
    476.5824781648738
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
        (
            self.differenced_data_,
            self.d_,
            self.model_,
            self.parameters_,
            self.aic_,
        ) = _auto_arima(self.data_)
        (
            self.constant_term_,
            self.p_,
            self.q_,
        ) = self.model_
        (self.c_, self.phi_, self.theta_) = _extract_params(
            self.parameters_, self.model_
        )
        (
            self.aic_,
            self.residuals_,
        ) = _arima_model(
            self.parameters_, _calc_arima, self.differenced_data_, self.model_
        )
        return self


def _auto_arima(data):
    """
    Prepare data for the AutoARIMA algorithm.

    This function checks if the data is stationary
    and applies differencing if necessary.
    """
    difference = 0
    while not kpss_test(data)[1]:
        data = np.diff(data, n=1)
        difference += 1
    include_c = 1 if difference == 0 else 0
    model_parameters = np.array(
        [
            [include_c, 2, 2],
            [include_c, 0, 0],
            [include_c, 1, 0],
            [include_c, 0, 1],
        ]
    )
    (
        differenced_data,
        best_model,
        best_points,
        best_score,
    ) = _auto_arma(data, model_parameters, 3)
    return (
        differenced_data,
        difference,
        best_model,
        best_points,
        best_score,
    )


def _auto_arma(differenced_data, inital_model_parameters, num_model_params=3):
    """
    Implement the Hyndman-Khandakar algorithm.

    For automatic ARIMA model selection.
    """
    best_score = -1
    best_model = inital_model_parameters[0]
    best_points = None
    for i in range(len(inital_model_parameters)):
        points, aic = nelder_mead(
            _arima_model_wrapper,
            np.sum(inital_model_parameters[i][:num_model_params]),
            differenced_data,
            inital_model_parameters[i],
        )
        if (aic < best_score) or (best_score == -1):
            best_score = aic
            best_model = inital_model_parameters[i]
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
                        _arima_model_wrapper,
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
        differenced_data,
        best_model,
        best_points,
        best_score,
    )
