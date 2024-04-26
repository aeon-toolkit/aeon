"""Feature based time series classifiers.

While a bit vague, the contents mostly consist of transformers that extract features
pipelined to a vector classifier.
"""

__all__ = [
    "Catch22Classifier",
    "SignatureClassifier",
    "SummaryClassifier",
    "TSFreshClassifier",
    "FreshPRINCEClassifier",
]

from aeon.classification.feature_based._catch22 import Catch22Classifier
from aeon.classification.feature_based._fresh_prince import FreshPRINCEClassifier
from aeon.classification.feature_based._signature_classifier import SignatureClassifier
from aeon.classification.feature_based._summary import SummaryClassifier
from aeon.classification.feature_based._tsfresh import TSFreshClassifier
