"""Test ETS."""

__maintainer__ = []
__all__ = []

import numpy as np
import pytest

from aeon.forecasting import ETS
from aeon.forecasting.stats._ets import _validate_parameter


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            dict(
                alpha=0.5,
                beta=0.3,
                gamma=0.4,
                phi=1,
                error_type="additive",
                trend_type="additive",
                seasonality_type="additive",
                seasonal_period=4,
            ),
            11.456563248800002,
        ),
        (
            dict(
                alpha=0.7,
                beta=0.6,
                gamma=0.1,
                phi=0.97,
                error_type="multiplicative",
                trend_type="additive",
                seasonality_type="additive",
                seasonal_period=4,
            ),
            15.507105356706465,
        ),
        (
            dict(
                alpha=0.4,
                beta=0.2,
                gamma=0.5,
                phi=0.8,
                error_type="additive",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            13.168538863095991,
        ),
        (
            dict(
                alpha=0.7,
                beta=0.5,
                gamma=0.2,
                phi=0.85,
                error_type="multiplicative",
                trend_type="multiplicative",
                seasonality_type="multiplicative",
                seasonal_period=4,
            ),
            15.223040987015944,
        ),
    ],
)
def test_ets_forecaster(params, expected):
    """Test ETS for multiple parameter combinations."""
    data = np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12])
    forecaster = ETS(**params)
    p = forecaster.forecast(data)
    assert np.isclose(p, expected)


def test_incorrect_parameters():
    """Test incorrect set up."""
    _validate_parameter(0, True)
    _validate_parameter(None, True)
    with pytest.raises(ValueError):
        _validate_parameter(0, False)
        _validate_parameter(None, True)
        _validate_parameter(10, False)
        _validate_parameter("Foo", True)
    forecaster = ETS()
    forecaster.horizon = 2
    data = np.array([3, 10, 12, 13, 12, 10, 12, 3, 10, 12, 13, 12, 10, 12])
    with pytest.raises(ValueError, match="Horizon is set >1, but"):
        forecaster.fit(data)
    forecaster = ETS()
    with pytest.raises(
        ValueError, match="This forecaster cannot be used with the " "direct strategy"
    ):
        forecaster.direct_forecast(data, prediction_horizon=6)
