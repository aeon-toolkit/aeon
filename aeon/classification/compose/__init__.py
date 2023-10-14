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
