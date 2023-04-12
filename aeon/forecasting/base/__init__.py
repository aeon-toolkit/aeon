# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Implements base classes for forecasting in aeon."""

__all__ = [
    "ForecastingHorizon",
    "BaseForecaster",
]

from aeon.forecasting.base._fh import ForecastingHorizon  # isort:skip
from aeon.forecasting.base._base import BaseForecaster  # isort:skip
