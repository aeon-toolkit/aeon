"""Compositions for classifiers."""

__all__ = [
    "ChannelEnsembleClassifier",
    "WeightedEnsembleClassifier",
    "ClassifierPipeline",
]

from aeon.classification.compose._channel_ensemble import ChannelEnsembleClassifier
from aeon.classification.compose._ensemble import WeightedEnsembleClassifier
from aeon.classification.compose._pipeline import ClassifierPipeline
