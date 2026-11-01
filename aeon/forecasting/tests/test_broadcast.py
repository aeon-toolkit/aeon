"""Tests for channel-wise forecast broadcasting."""

import numpy as np
import pytest
from sklearn.base import clone

from aeon.forecasting import (
    BroadcastForecaster,
    NaiveForecaster,
    RegressionForecaster,
)
from aeon.forecasting.base import BaseForecaster


class _VectorForecaster(BaseForecaster):
    """Invalid wrapped forecaster that returns a vector for one channel."""

    def __init__(self):
        super().__init__(horizon=1, axis=1)

    def _fit(self, y, exog=None):
        self.forecast_ = np.array([1.0, 2.0])
        return self

    def _predict(self, y, exog=None):
        return np.array([1.0, 2.0])


def test_broadcast_forecaster_forecasts_each_channel_independently():
    """Broadcast a naive forecaster over multivariate input."""
    y = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    forecaster = BroadcastForecaster(NaiveForecaster(strategy="last"))
    pred = forecaster.forecast(y)

    np.testing.assert_array_equal(pred, np.array([3.0, 30.0]))
    np.testing.assert_array_equal(forecaster.forecast_, np.array([3.0, 30.0]))
    assert len(forecaster.forecasters_) == 2
    assert forecaster.forecasters_[0] is not forecaster.forecaster


def test_broadcast_forecaster_fit_predict_uses_predict_context():
    """Predict one value per channel using the context supplied to predict."""
    y_train = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    y_predict = np.array([[4.0, 5.0], [40.0, 50.0]])

    forecaster = BroadcastForecaster(NaiveForecaster(strategy="last")).fit(y_train)
    pred = forecaster.predict(y_predict)

    np.testing.assert_array_equal(pred, np.array([5.0, 50.0]))


def test_broadcast_forecaster_returns_scalar_for_univariate_input():
    """Return a scalar when the original input was univariate."""
    y = np.array([1.0, 2.0, 3.0])

    forecaster = BroadcastForecaster(NaiveForecaster(strategy="last"))
    pred = forecaster.forecast(y)

    assert isinstance(pred, float)
    assert pred == 3.0


def test_broadcast_forecaster_respects_input_axis():
    """Handle multivariate input with time points on axis 0."""
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    forecaster = BroadcastForecaster(NaiveForecaster(strategy="last"))
    pred = forecaster.forecast(y, axis=0)

    np.testing.assert_array_equal(pred, np.array([3.0, 30.0]))


@pytest.mark.parametrize("replace_forecaster", [False, True])
def test_broadcast_forecaster_forwards_exog_to_channel_forecasters(replace_forecaster):
    """Exogenous predictions match separate fits, including after replacement."""
    n_timepoints = 12
    time = np.arange(n_timepoints, dtype=float)
    y_train = np.array([time**2 + time, time**2 - time])
    fit_exog = (time**2)[None, :]
    predict_exog = np.array([[float(n_timepoints**2)]])
    model = RegressionForecaster(window=2)
    forecaster = BroadcastForecaster(NaiveForecaster() if replace_forecaster else model)
    if replace_forecaster:
        forecaster.set_params(forecaster=model)
    forecaster.fit(y_train, exog=fit_exog)

    expected = [
        clone(model).fit(channel, exog=fit_exog).predict(channel, exog=predict_exog)
        for channel in y_train
    ]
    np.testing.assert_allclose(forecaster.predict(y_train, exog=predict_exog), expected)


@pytest.mark.parametrize("method", ["fit", "forecast"])
def test_broadcast_forecaster_uses_updated_horizon(method):
    """Nested horizon updates determine the forecast for each channel."""
    y = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    horizon = 3
    forecaster = BroadcastForecaster(NaiveForecaster(strategy="drift"))
    forecaster.set_params(forecaster__horizon=horizon)
    getattr(forecaster, method)(y)

    expected = y[:, -1] + horizon * (y[:, -1] - y[:, 0]) / (y.shape[1] - 1)
    np.testing.assert_allclose(forecaster.forecast_, expected)
    np.testing.assert_allclose(forecaster.predict(y), expected)


def test_broadcast_forecaster_rejects_invalid_forecaster():
    """Reject non-forecaster wrapped estimators."""
    with pytest.raises(TypeError, match="BaseForecaster"):
        BroadcastForecaster(object())


def test_broadcast_forecaster_rejects_channel_mismatch_in_predict():
    """Reject predict input with a different number of channels from fit."""
    y_train = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    y_predict = np.array([[4.0, 5.0]])

    forecaster = BroadcastForecaster(NaiveForecaster(strategy="last")).fit(y_train)

    with pytest.raises(ValueError, match="number of channels"):
        forecaster.predict(y_predict)


def test_broadcast_forecaster_rejects_non_scalar_channel_prediction():
    """Reject wrapped forecasters that return more than one value per channel."""
    y = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    forecaster = BroadcastForecaster(_VectorForecaster())

    with pytest.raises(ValueError, match="must return a scalar"):
        forecaster.forecast(y)


def test_broadcast_forecaster_can_be_sklearn_cloned():
    """Check constructor parameters are clone-compatible."""
    forecaster = BroadcastForecaster(NaiveForecaster(strategy="last"))

    cloned = clone(forecaster)

    assert isinstance(cloned, BroadcastForecaster)
    assert isinstance(cloned.forecaster, NaiveForecaster)
