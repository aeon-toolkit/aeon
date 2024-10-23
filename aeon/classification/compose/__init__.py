"""Compositions for classifiers."""

__all__ = [
    "ClassifierChannelEnsemble",
    "ClassifierEnsemble",
    "ClassifierPipeline",
]

from aeon.classification.compose._channel_ensemble import ClassifierChannelEnsemble
from aeon.classification.compose._ensemble import ClassifierEnsemble
from aeon.classification.compose._pipeline import ClassifierPipeline
