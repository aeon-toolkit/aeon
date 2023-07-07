# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Implement interval-based time series regression estimators."""

__all__ = [
    "RandomIntervalRegressor",
    "TimeSeriesForestRegressor",
]

from aeon.regression.interval_based._interval_pipelines import RandomIntervalRegressor
from aeon.regression.interval_based._tsf import TimeSeriesForestRegressor
