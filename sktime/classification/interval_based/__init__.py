# -*- coding: utf-8 -*-
"""Interval based time series classifiers."""
__all__ = [
    "TimeSeriesForestClassifier",
    "RandomIntervalSpectralEnsemble",
    "SupervisedTimeSeriesForest",
    "CanonicalIntervalForest",
    "DrCIF",
    "RandomIntervalClassifier",
]

from sktime.classification.interval_based._cif import CanonicalIntervalForest
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.interval_based._random_interval_classifier import (
    RandomIntervalClassifier,
)
from sktime.classification.interval_based._rise import RandomIntervalSpectralEnsemble
from sktime.classification.interval_based._stsf import SupervisedTimeSeriesForest
from sktime.classification.interval_based._tsf import TimeSeriesForestClassifier
