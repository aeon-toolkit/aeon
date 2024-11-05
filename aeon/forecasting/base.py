"""BaseForecaster class.

A simplified first base class for foreacasting models. The focus here is on a
specific form of forecasting: longer series, long winodws and single step forecasting.

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon-admin/pull/14

"""

from abc import abstractmethod

from aeon.base import BaseSeriesEstimator


class BaseForecaster(BaseSeriesEstimator):
    """
    Abstract base class for time series forecasters.

    The base forecaster specifies the methods and method signatures that all
    forecasters have to implement. Attributes with an underscore suffix are set in the
    method fit.

    Parameters
    ----------
    horizon : int, default =1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "fit_is_empty": False,
    }

    def __init__(self, horizon=1, axis=1):
        self.horizon = horizon
        super().__init__(axis)

    def fit(self, y, exog=None):
        """Fit forecaster to series y.

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
            Fitted BaseForecaster.
        """
        # Validate y

        # Convert if necessary
        y = self._preprocess_series(y, axis=self.axis, store_metadata=False)
        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")
        # Validate exog
        self.is_fitted = True
        return self._fit(y, exog)

    @abstractmethod
    def _fit(self, y, exog=None): ...

    def predict(self, y=None, exog=None):
        """Predict the next horizon steps ahead.

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
        if y is not None:
            y = self._preprocess_series(y, axis=self.axis, store_metadata=False)
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before predicting")
        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")
        # Validate exog
        self.is_fitted = True
        return self._predict(y, exog)

    @abstractmethod
    def _predict(self, y=None, exog=None): ...

    def forecast(self, y, X=None):
        """

        Forecast basically fit_predict.

        Returns
        -------
        np.ndarray
            single prediction directly after the last point in X.
        """
        y = self._preprocess_series(y, axis=self.axis, store_metadata=False)
        return self._forecast(y, X)

    @abstractmethod
    def _forecast(self, y=None, exog=None): ...
