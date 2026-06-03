import numpy as np

from aeon.base._base import _clone_estimator
from aeon.forecasting.base import BaseForecaster


class BroadcastForecaster(BaseForecaster):
    """A wrapper for multtivariate forecasting.

    Applies a given univariate capable forecaster independently to each channel of a
    multivariate series.

    Parameters
    ----------
    forecaster : BaseForecaster
        An aeon-compatible univariate forecaster instance.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:horizon": True,
        "capability:exogenous": False,
        "fit_is_empty": False,
        "y_inner_type": "np.ndarray",
    }

    def __init__(self, forecaster: BaseForecaster):
        self.forecaster = forecaster
        self.forecasters_ = []
        super().__init__(horizon=forecaster.horizon, axis=forecaster.axis)
        # match exogenous capability to wrapped forecaster
        self.set_tags(
            **{"capability:exogenous": forecaster.get_tag("capability:exogenous")}
        )

    def _fit(self, y, exog=None):
        """Fit one forecaster per channel."""
        n_channels = y.shape[0]  # (n_channels, n_timepoints) after preprocessing
        self.forecasters_ = []
        self.forecast_ = np.zeros(n_channels)
        for i in range(n_channels):
            f = _clone_estimator(self.forecaster)
            f.horizon = self.horizon
            f.fit(y[i], exog)
            self.forecasters_.append(f)
            self.forecast_[i] = f.forecast_
        if n_channels == 1:
            self.forecast_f.forecast_[0]
        return self

    def _predict(self, y, exog=None):
        """Predict one ahead for each channel independently."""
        n_channels = y.shape[0]
        preds = np.zeros(n_channels, dtype=float)
        for c in range(n_channels):
            preds[c] = self.forecasters_[c].predict(y[c], exog=None)
        return preds
