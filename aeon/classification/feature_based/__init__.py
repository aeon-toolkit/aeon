# -*- coding: utf-8 -*-
"""Feature based time series classifiers.

While a bit vague, the contents mostly consist of transformers that extract features
pipelined to a vector classifier.
"""

__all__ = [
    "Catch22Classifier",
    "MatrixProfileClassifier",
    "SignatureClassifier",
    "SummaryClassifier",
    "TSFreshClassifier",
    "FreshPRINCE",
]

from aeon.classification.feature_based._catch22_classifier import Catch22Classifier
from aeon.classification.feature_based._fresh_prince import FreshPRINCE
from aeon.classification.feature_based._matrix_profile_classifier import (
    MatrixProfileClassifier,
)
from aeon.classification.feature_based._signature_classifier import SignatureClassifier
from aeon.classification.feature_based._summary_classifier import SummaryClassifier
from aeon.classification.feature_based._tsfresh_classifier import TSFreshClassifier
