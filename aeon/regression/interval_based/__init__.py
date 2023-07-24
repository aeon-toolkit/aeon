# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Implement interval-based time series regression estimators."""

__all__ = [
    "CanonicalIntervalForestRegressor",
    "DrCIFRegressor",
    "IntervalForestRegressor",
    "RandomIntervalRegressor",
    "RandomIntervalSpectralEnsembleRegressor",
    "TimeSeriesForestRegressor",
]

from aeon.regression.interval_based._cif import CanonicalIntervalForestRegressor
from aeon.regression.interval_based._drcif import DrCIFRegressor
from aeon.regression.interval_based._interval_forest import IntervalForestRegressor
from aeon.regression.interval_based._interval_pipelines import RandomIntervalRegressor
from aeon.regression.interval_based._rise import RandomIntervalSpectralEnsembleRegressor
from aeon.regression.interval_based._tsf import TimeSeriesForestRegressor
