# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Interval-based time series classifiers."""

__all__ = [
    "CanonicalIntervalForestClassifier",
    "DrCIFClassifier",
    "IntervalForestClassifier",
    "RandomIntervalClassifier",
    "SupervisedIntervalClassifier",
    "RandomIntervalSpectralEnsembleClassifier",
    "RSTSF",
    "SupervisedTimeSeriesForest",
    "TimeSeriesForestClassifier",
]

from aeon.classification.interval_based._cif import CanonicalIntervalForestClassifier
from aeon.classification.interval_based._drcif import DrCIFClassifier
from aeon.classification.interval_based._interval_forest import IntervalForestClassifier
from aeon.classification.interval_based._interval_pipelines import (
    RandomIntervalClassifier,
    SupervisedIntervalClassifier,
)
from aeon.classification.interval_based._rise import (
    RandomIntervalSpectralEnsembleClassifier,
)
from aeon.classification.interval_based._rstsf import RSTSF
from aeon.classification.interval_based._stsf import SupervisedTimeSeriesForest
from aeon.classification.interval_based._tsf import TimeSeriesForestClassifier
