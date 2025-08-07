"""Test the TAR forecaster."""

import numpy as np
import pytest

from aeon.datasets import load_airline
from aeon.forecasting.stats._tar import TAR  # adjust if it's in another module


@pytest.fixture
def airline_series():
    """Fixture to load a simple test time series."""
    return load_airline().squeeze()


def test_fit_forecast_output(airline_series):
    """Test that fit runs and forecast_ is a float."""
    model = TAR(ar_order=2, delay=1)
    model.fit(airline_series)
    assert isinstance(model.forecast_, float)
    assert np.isfinite(model.forecast_)


def test_predict_output(airline_series):
    """Test that _predict returns a float for valid input."""
    model = TAR(ar_order=2, delay=1)
    model.fit(airline_series)
    pred = model._predict(airline_series[:-1])
    assert isinstance(pred, float)
    assert np.isfinite(pred)


def test_params_structure(airline_series):
    """Test that params_ dict has correct structure after fitting."""
    model = TAR(ar_order=2, delay=1)
    model.fit(airline_series)
    params = model.params_

    assert "threshold" in params
    assert "regime_1" in params
    assert "regime_2" in params

    for regime in ["regime_1", "regime_2"]:
        assert "intercept" in params[regime]
        assert "coefficients" in params[regime]
        assert isinstance(params[regime]["coefficients"], list)


def test_custom_threshold(airline_series):
    """Test that a user-defined threshold is respected."""
    threshold = 400.0
    model = TAR(ar_order=2, delay=1, threshold=threshold)
    model.fit(airline_series)
    assert model.threshold_ == threshold


def test_forecast_changes_with_series(airline_series):
    """Test that modifying the series changes the forecast."""
    model1 = TAR(ar_order=2, delay=1)
    model2 = TAR(ar_order=2, delay=1)

    model1.fit(airline_series)
    modified_series = airline_series.copy()
    modified_series[-1] += 100
    model2.fit(modified_series)

    assert not np.isclose(model1.forecast_, model2.forecast_)
