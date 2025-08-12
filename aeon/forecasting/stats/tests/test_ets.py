"""Test ETS."""

import numpy as np
import pytest

from aeon.forecasting.stats._ets import ETS, _validate_parameter


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            dict(
                error_type="additive",
                trend_type="additive",
                seasonality_type="additive",
                seasonal_period=4,
            ),
            11.150969310377484,
        ),
        (
            dict(
                error_type="multiplicative",
                trend_type="additive",
                seasonality_type="additive",
                seasonal_period=4,
            ),
            11.15096931037748,
        ),
        (
            dict(
                error_type="additive",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            14.075007324719092,
        ),
        (
            dict(
                error_type="multiplicative",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            14.075007324719058,
        ),
    ],
)
def test_ets_forecaster(params, expected):
    """Test ETS for multiple parameter combinations."""
    data = np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12])
    forecaster = ETS(**params)
    p = forecaster.forecast(data)
    assert np.isclose(p, expected)


@pytest.mark.parametrize(
    "value, allow_none, should_raise",
    [
        (0, True, False),  # valid
        (None, True, False),  # valid
        (0, False, True),  # invalid
        (10, False, True),  # invalid
        ("Foo", True, True),  # invalid
    ],
)
def test_validate_parameter_cases(value, allow_none, should_raise):
    """Test _validate_parameter with valid and invalid inputs."""
    if should_raise:
        with pytest.raises(ValueError):
            _validate_parameter(value, allow_none)
    else:
        _validate_parameter(value, allow_none)


def test_ets_raises_on_horizon_greater_than_one():
    """Test that ETS.fit raises ValueError when horizon > 1."""
    forecaster = ETS()
    forecaster.horizon = 2
    data = np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12])
    with pytest.raises(ValueError, match="Horizon is set >1, but"):
        forecaster.fit(data)


def test_ets_iterative_forecast():
    """Test the ETS.iterative_forecast method produces correct number of forecasts."""
    forecaster = ETS(
        trend_type="additive", seasonality_type="additive", seasonal_period=4
    )
    y = np.array([10, 12, 14, 13, 15, 16, 18, 19, 20, 21, 22, 23])
    h = 5

    preds = forecaster.iterative_forecast(y, prediction_horizon=h)

    assert isinstance(preds, np.ndarray), "Output should be a NumPy array"
    assert preds.shape == (h,), f"Expected output shape {(h,)}, got {preds.shape}"
    assert np.all(np.isfinite(preds)), "All forecast values should be finite"

    # Optional: check that the first prediction equals forecast_ from .fit()
    forecaster.fit(y)
    assert np.isclose(
        preds[0], forecaster.forecast_, atol=1e-6
    ), "First forecast should match forecast_"
    forecaster = ETS(trend_type=None)
    forecaster._fit(y)
    assert forecaster._trend_type == 0
