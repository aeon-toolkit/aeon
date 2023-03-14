# -*- coding: utf-8 -*-
"""Utility estimator classes for testing and debugging."""

__author__ = ["ltsaprounis"]

__all__ = [
    "MockForecaster",
    "MockUnivariateForecasterLogger",
    "make_mock_estimator",
    "construct_dispatch",
]

from aeon.utils.estimators._base import make_mock_estimator
from aeon.utils.estimators._forecasters import (
    MockForecaster,
    MockUnivariateForecasterLogger,
)
from aeon.utils.estimators.dispatch import construct_dispatch
