"""Mock forecasters for testing and debugging."""

__author__ = ["ltsaprounis"]

__all__ = [
    "MockForecaster",
    "MockUnivariateForecasterLogger",
    "make_mock_estimator",
]

from aeon.utils.estimators._mock_forecasters import (
    MockForecaster,
    MockUnivariateForecasterLogger,
    make_mock_estimator,
)
