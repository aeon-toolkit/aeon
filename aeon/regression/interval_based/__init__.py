# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)-
"""Interval based time series reggressors."""

__all__ = ["IntervalForestRegressor", "TimeSeriesForestRegressor"]

from aeon.regression.interval_based._interval_forest import IntervalForestRegressor
from aeon.regression.interval_based._tsf import TimeSeriesForestRegressor
