"""DummyForecaster always predicts the last value seen in training."""

from aeon.forecasting.base import BaseForecaster


class DummyForecaster(BaseForecaster):
    """Dummy forecaster always predicts the last value seen in training."""

    def __init__(self):
        """Initialize ``DummyForecaster``."""
        self.last_value_ = None
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        """Fit ``DummyForecaster``.
        
        Parameters
        ----------
        y : array-like
            Target values to fit the forecaster.
        exog : array-like, optional
            Exogenous variables (default is ``None``).
        """
        y = y.squeeze()
        self.last_value_ = y[-1]
        return self

    def _predict(self, y=None, exog=None):
        """Predict using ``DummyForecaster``.
        
        Parameters
        ----------
        y : array-like, optional
            Input data (default is ``None``).
        exog : array-like, optional
            Exogenous variables (default is ``None``).

        Returns
        -------
        float
            The last observed value.
        """
        return self.last_value_

    def _forecast(self, y, exog=None):
        """Forecast using ``DummyForecaster``.
        
        Parameters
        ----------
        y : array-like
            Input data for forecasting.
        exog : array-like, optional
            Exogenous variables (default is ``None``).

        Returns
        -------
        float
            Forecasted value based on the last observed value.
        """
        y = y.squeeze()
        return y[-1]
