# -*- coding: utf-8 -*-
"""Compositions for classifiers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "fkiraly"]
__all__ = [
    "ClassifierPipeline",
    "ComposableTimeSeriesForestClassifier",
    "ColumnEnsembleClassifier",
    "SklearnClassifierPipeline",
    "WeightedEnsembleClassifier",
]

from aeon.classification.compose._column_ensemble import ColumnEnsembleClassifier
from aeon.classification.compose._ensemble import (
    ComposableTimeSeriesForestClassifier,
    WeightedEnsembleClassifier,
)
from aeon.classification.compose._pipeline import (
    ClassifierPipeline,
    SklearnClassifierPipeline,
)
