"""Initialization for aeon forecasting deep learning module."""

__all__ = [
    "BaseDeepForecaster",
    "TCNForecaster",
]

from aeon.forecasting.deep_learning._tcn import TCNForecaster
from aeon.forecasting.deep_learning.base import BaseDeepForecaster
