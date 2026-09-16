"""Tests for channel-wise forecast broadcasting."""

import numpy as np
import pytest
from sklearn.base import clone

from aeon.forecasting import BroadcastForecaster, NaiveForecaster
from aeon.forecasting.base import BaseForecaster


class _ExogRecorderForecaster(BaseForecaster):
    """Small forecaster that records exog and adds it to the last value."""

    _tags = {"capability:exogenous": True}

    def __init__(self, horizon=1):
        super().__init__(horizon=horizon, axis=1)

    def _fit(self, y, exog=None):
        self.fit_exog_shape_ = None if exog is None else exog.shape
        self.forecast_ = float(y.squeeze()[-1] + np.asarray(exog).sum())
        return self

    def _predict(self, y, exog=None):
        self.predict_exog_shape_ = None if exog is None else exog.shape
        return float(y.squeeze()[-1] + np.asarray(exog).sum())


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


def test_broadcast_forecaster_forwards_exog_to_channel_forecasters():
    """Forward fit and predict exog to each wrapped channel forecaster."""
    y_train = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    y_predict = np.array([[4.0, 5.0], [40.0, 50.0]])
    fit_exog = np.array([[1.0, 2.0, 3.0]])
    predict_exog = np.array([[10.0]])

    forecaster = BroadcastForecaster(_ExogRecorderForecaster()).fit(
        y_train, exog=fit_exog
    )
    pred = forecaster.predict(y_predict, exog=predict_exog)

    np.testing.assert_array_equal(pred, np.array([15.0, 60.0]))
    for channel_forecaster in forecaster.forecasters_:
        assert channel_forecaster.fit_exog_shape_ == fit_exog.shape
        assert channel_forecaster.predict_exog_shape_ == predict_exog.shape


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
