# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Interval-based time series classifiers."""

__all__ = [
    "CanonicalIntervalForest",
    "DrCIF",
    "IntervalForestClassifier",
    "RandomIntervalClassifier",
    "SupervisedIntervalClassifier",
    "RandomIntervalSpectralEnsemble",
    "RSTSF",
    "SupervisedTimeSeriesForest",
    "TimeSeriesForestClassifier",
]

from aeon.classification.interval_based._cif import CanonicalIntervalForest
from aeon.classification.interval_based._drcif import DrCIF
from aeon.classification.interval_based._interval_forest import IntervalForestClassifier
from aeon.classification.interval_based._interval_pipelines import (
    RandomIntervalClassifier,
    SupervisedIntervalClassifier,
)
from aeon.classification.interval_based._rise import RandomIntervalSpectralEnsemble
from aeon.classification.interval_based._rstsf import RSTSF
from aeon.classification.interval_based._stsf import SupervisedTimeSeriesForest
from aeon.classification.interval_based._tsf import TimeSeriesForestClassifier
