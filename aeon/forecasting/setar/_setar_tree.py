import numpy as np

from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.setar._setar import SETARForecaster


class SETARTreeForecaster(BaseForecaster):
    """
    SETAR-Tree forecaster (baseline implementation).

    Trains a global SETAR model across a collection of time series.
    """

    _tags = {
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:global_forecasting": True,
    }

    def __init__(self, lags=1, threshold_lag=1):
        self.lags = lags
        self.threshold_lag = threshold_lag
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        # y is expected to be a collection (list-like)
        self.models_ = []

        for series in y:
            model = SETARForecaster(
                lags=self.lags, threshold_lag=self.threshold_lag
            )
            model.fit(series)
            self.models_.append(model)

        return self

    def _predict(self, fh, X=None):
        preds = np.array([m.predict(fh) for m in self.models_])
        return preds.mean(axis=0)
