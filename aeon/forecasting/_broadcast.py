"""Broadcast a univariate forecaster over channels of a multivariate series."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["BroadcastForecaster"]

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.forecasting.base import BaseForecaster


class BroadcastForecaster(BaseForecaster):
    """Apply a univariate forecaster independently to each channel.

    ``BroadcastForecaster`` is a lightweight wrapper for forecasters that work on a
    single target series. It clones the supplied forecaster once per channel, fits
    each clone to one channel of ``y``, and returns one prediction per channel for
    multivariate input. For univariate input it returns a scalar, matching the
    wrapped forecaster's usual output shape.

    Parameters
    ----------
    forecaster : BaseForecaster
        Aeon-compatible forecaster to clone and fit independently per channel.

    Attributes
    ----------
    forecasters_ : list of BaseForecaster
        Fitted channel-specific clones of ``forecaster``.
    forecast_ : float or np.ndarray
        In-sample one-step forecast after fitting. A scalar for univariate data and
        an array of shape ``(n_channels,)`` for multivariate data.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.forecasting import NaiveForecaster
    >>> from aeon.forecasting import BroadcastForecaster
    >>> y = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    >>> forecaster = BroadcastForecaster(NaiveForecaster(strategy="last"))
    >>> forecaster.forecast(y)
    array([ 3., 30.])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:horizon": True,
        "capability:exogenous": False,
        "fit_is_empty": False,
        "X_inner_type": "np.ndarray",
    }

    def __init__(self, forecaster):
        if not isinstance(forecaster, BaseForecaster):
            raise TypeError("forecaster must be an instance of BaseForecaster.")
        self.forecaster = forecaster
        super().__init__(horizon=forecaster.horizon, axis=1)
        self.set_tags(
            **{
                "capability:exogenous": forecaster.get_tag("capability:exogenous"),
                "capability:horizon": forecaster.get_tag("capability:horizon"),
                "capability:missing_values": forecaster.get_tag(
                    "capability:missing_values"
                ),
            }
        )

    def _fit(self, y, exog=None):
        """Fit one clone of the wrapped forecaster per channel."""
        self.forecasters_ = []
        forecasts = np.empty(y.shape[0], dtype=float)

        for i in range(y.shape[0]):
            forecaster = _clone_estimator(self.forecaster)
            forecaster.horizon = self.horizon
            channel_forecast = forecaster.forecast(y[i], exog=exog)
            self.forecasters_.append(forecaster)
            forecasts[i] = self._as_scalar_prediction(channel_forecast)

        self.forecast_ = self._format_output(forecasts)
        return self

    def _predict(self, y, exog=None):
        """Predict independently for each channel using fitted channel clones."""
        if y.shape[0] != len(self.forecasters_):
            raise ValueError(
                "The number of channels in predict does not match fit. "
                f"Saw {y.shape[0]}, expected {len(self.forecasters_)}."
            )

        predictions = np.empty(y.shape[0], dtype=float)
        for i, forecaster in enumerate(self.forecasters_):
            predictions[i] = self._as_scalar_prediction(
                forecaster.predict(y[i], exog=exog)
            )
        return self._format_output(predictions)

    def _forecast(self, y, exog=None):
        """Fit on ``y`` and return the fitted channel forecasts."""
        self._fit(y, exog=exog)
        return self.forecast_

    def _format_output(self, values):
        """Return scalar output for univariate input and vector output otherwise."""
        if not self.metadata_.get("multivariate", False):
            return float(values[0])
        return values

    @staticmethod
    def _as_scalar_prediction(prediction):
        """Convert a wrapped forecaster prediction to a scalar float."""
        prediction = np.asarray(prediction)
        if prediction.size != 1:
            raise ValueError(
                "The wrapped forecaster must return a scalar prediction for each "
                f"channel, but returned shape {prediction.shape}."
            )
        return float(prediction.reshape(-1)[0])

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from aeon.forecasting import NaiveForecaster

        return {"forecaster": NaiveForecaster(strategy="last")}
