"""AutoETS class.

Extends the ETSForecaster to automatically calculate the smoothing parameters.
"""

__maintainer__ = ["alexbanwell1"]
__all__ = ["AutoETS"]
import numpy as np

from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.stats._ets import ETS
from aeon.forecasting.utils._nelder_mead import nelder_mead
from aeon.forecasting.utils._seasonality import calc_seasonal_period


class AutoETS(BaseForecaster):
    """Automatic Exponential Smoothing forecaster.

    An implementation of the exponential smoothing statistics forecasting algorithm.
    Chooses betweek additive and multiplicative error models,
    None, additive and multiplicative (including damped) trend and
    None, additive and multiplicative seasonality[1]_.

    Parameters
    ----------
    horizon : int, default = 1
        The horizon to forecast to.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos,
        Forecasting: Principles and Practice. Melbourne, Australia: OTexts, 2014.

    Examples
    --------
    >>> from aeon.forecasting import AutoETSForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AutoETSForecaster()
    >>> forecaster.fit(y)
    AutoETSForecaster()
    >>> forecaster.predict()
    array([407.74740434])
    """

    def __init__(self):
        self.error_type_ = 0
        self.trend_type_ = 0
        self.seasonality_type_ = 0
        self.seasonal_period_ = 0
        self.wrapped_model_ = None
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit Auto Exponential Smoothing forecaster to series y.

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
            Fitted AutoETS.
        """
        data = y.squeeze()
        (
            self.error_type_,
            self.trend_type_,
            self.seasonality_type_,
            self.seasonal_period_,
        ) = auto_ets(data)
        self.wrapped_model_ = ETS(
            self.error_type_,
            self.trend_type_,
            self.seasonality_type_,
            self.seasonal_period_,
        )
        self.wrapped_model_.fit(y, exog)
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
        return self.wrapped_model_.predict(y, exog)

    def iterative_forecast(self, y, prediction_horizon):
        """Forecast with ETS specific iterative method.

        Overrides the base class iterative_forecast to avoid refitting on each step.
        This simply rolls the ETS model forward
        """
        return self.wrapped_model_.iterative_forecast(y, prediction_horizon)


def auto_ets(data):
    """Calculate model parameters based on the internal nelder-mead implementation."""
    seasonal_period = calc_seasonal_period(data)
    lowest_aic = -1
    best_model = None
    for error_type in range(1, 3):
        for trend_type in range(0, 3):
            for seasonality_type in range(0, 2 * (seasonal_period != 1) + 1):
                model_seasonal_period = seasonal_period
                if seasonal_period < 1 or seasonality_type == 0:
                    model_seasonal_period = 1
                model = np.array(
                    [
                        error_type,
                        trend_type,
                        seasonality_type,
                        model_seasonal_period,
                    ],
                    dtype=np.int32,
                )
                try:
                    (_, aic) = nelder_mead(
                        1,
                        1 + 2 * (trend_type != 0) + (seasonality_type != 0),
                        data,
                        model,
                    )
                except ZeroDivisionError:
                    continue
                if lowest_aic == -1 or lowest_aic > aic:
                    lowest_aic = aic
                    best_model = (
                        error_type,
                        trend_type,
                        seasonality_type,
                        model_seasonal_period,
                    )
    return best_model
