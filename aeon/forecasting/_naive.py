"""ETSForecaster class.

An implementation of the exponential smoothing statistics forecasting algorithm.
Implements additive and multiplicative error models,
None, additive and multiplicative (including damped) trend and
None, additive and mutliplicative seasonality

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon/pull/2244/

"""

__maintainer__ = []
__all__ = ["NaiveForecaster"]

import numpy as np

from aeon.forecasting.base import BaseForecaster

NONE = 0
ADDITIVE = 1
MULTIPLICATIVE = 2


class NaiveForecaster(BaseForecaster):
    """Naive forecaster.

    Forecasts future values as the last observed value.

    Parameters
    ----------
    horizon : int, default = 1
        The number of steps ahead to forecast.

    Examples
    --------
    >>> from aeon.forecasting import NaiveForecaster
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster()
    >>> forecaster.fit(y)
    NaiveForecaster()
    >>> forecaster.predict()
    366.90200486015596
    """

    def __init__(
        self,
        horizon=1,
    ):
        self.last_value_ = None
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        """Fit Naive forecaster to series y.

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
            Fitted NaiveForecaster.
        """
        self.last_value_ = y[0][-1]
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
        if y is None:
            return np.array([self.last_value_])
        else:
            return np.insert(y, 0, self.last_value_)[:-1]
