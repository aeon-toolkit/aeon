# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Compositions for classifiers."""

__author__ = ["mloning", "fkiraly"]
__all__ = [
    "ChannelEnsembleClassifier",
    "WeightedEnsembleClassifier",
    "ClassifierPipeline",
    "SklearnClassifierPipeline",
]

from aeon.classification.compose._channel_ensemble import ChannelEnsembleClassifier
from aeon.classification.compose._ensemble import WeightedEnsembleClassifier
from aeon.classification.compose._pipeline import (
    ClassifierPipeline,
    SklearnClassifierPipeline,
)
