"""Implements base classes for forecasting in aeon."""

__all__ = [
    "ForecastingHorizon",
    "BaseForecaster",
]

from aeon.forecasting.base._base import BaseForecaster
from aeon.forecasting.base._fh import ForecastingHorizon
