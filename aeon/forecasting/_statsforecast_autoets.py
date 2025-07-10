"""StatsforecastAutoETS class.

Wraps statsforecast AutoETS model for forecasting.

"""

__maintainer__ = []
__all__ = ["StatsForecastAutoETSForecaster"]

from statsforecast.models import AutoETS

from aeon.forecasting._utils import calc_seasonal_period
from aeon.forecasting.base import BaseForecaster


class StatsForecastAutoETSForecaster(BaseForecaster):
    """Automatic Exponential Smoothing forecaster from statsforecast.

    Parameters
    ----------
    horizon : int, default = 1
        The horizon to forecast to.
    """

    def __init__(
        self,
        horizon=1,
    ):
        self.model_ = None
        super().__init__(horizon=horizon, axis=1)

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
            Fitted AutoETSForecaster.
        """
        data = y.squeeze()
        season_length = calc_seasonal_period(data)
        self.model_ = AutoETS(season_length=season_length)
        self.model_.fit(data)
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
        return self.model_.predict(self.horizon, exog)["mean"][0]
