"""Test ETS."""

__maintainer__ = []
__all__ = []

import numpy as np
import pytest

from aeon.forecasting import ETSForecaster
from aeon.forecasting._ets import _validate_parameter


def test_ets_forecaster_additive():
    """TestETSForecaster."""
    data = np.array(
        [3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]
    )  # Sample seasonal data
    forecaster = ETSForecaster(
        alpha=0.5,
        beta=0.3,
        gamma=0.4,
        phi=1,
        horizon=1,
        error_type="additive",
        trend_type="additive",
        seasonality_type="additive",
        seasonal_period=4,
    )
    forecaster.fit(data)
    p = forecaster.predict()
    assert np.isclose(p, 9.191190608800001)


def test_ets_forecaster_mult_error():
    """TestETSForecaster."""
    data = np.array(
        [3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]
    )  # Sample seasonal data
    forecaster = ETSForecaster(
        alpha=0.7,
        beta=0.6,
        gamma=0.1,
        phi=0.97,
        horizon=1,
        error_type="multiplicative",
        trend_type="additive",
        seasonality_type="additive",
        seasonal_period=4,
    )
    forecaster.fit(data)
    p = forecaster.predict()
    assert np.isclose(p, 16.20176819429869)


def test_ets_forecaster_mult_compnents():
    """TestETSForecaster."""
    data = np.array(
        [3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]
    )  # Sample seasonal data
    forecaster = ETSForecaster(
        alpha=0.4,
        beta=0.2,
        gamma=0.5,
        phi=0.8,
        horizon=1,
        error_type="additive",
        trend_type="multiplicative",
        seasonality_type="multiplicative",
        seasonal_period=4,
    )
    forecaster.fit(data)
    p = forecaster.predict()
    assert np.isclose(p, 12.301259229712382)


def test_ets_forecaster_multiplicative():
    """TestETSForecaster."""
    data = np.array(
        [3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12]
    )  # Sample seasonal data
    forecaster = ETSForecaster(
        alpha=0.7,
        beta=0.5,
        gamma=0.2,
        phi=0.85,
        horizon=1,
        error_type="multiplicative",
        trend_type="multiplicative",
        seasonality_type="multiplicative",
        seasonal_period=4,
    )
    forecaster.fit(data)
    p = forecaster.predict()
    assert np.isclose(p, 16.811888294476528)


def test_incorrect_parameters():
    _validate_parameter(0, True)
    _validate_parameter(None, True)
    with pytest.raises(ValueError):
        _validate_parameter(0, False)
        _validate_parameter(None, True)
        _validate_parameter(10, False)
        _validate_parameter("Foo", True)
