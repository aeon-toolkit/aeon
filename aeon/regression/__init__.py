"""Regression Base."""

__all__ = [
    "BaseRegressor",
    "DummyRegressor",
]

from aeon.regression._dummy import DummyRegressor
from aeon.regression.base import BaseRegressor
