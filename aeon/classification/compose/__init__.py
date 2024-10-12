"""Compositions for classifiers."""

__all__ = [
    "ChannelEnsembleClassifier",
    "ClassifierPipeline",
    "ClassifierEnsemble",
]

from aeon.classification.compose._channel_ensemble import ChannelEnsembleClassifier
from aeon.classification.compose._ensemble import ClassifierEnsemble
from aeon.classification.compose._pipeline import ClassifierPipeline
