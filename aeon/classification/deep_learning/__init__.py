# -*- coding: utf-8 -*-
"""Deep learning based classifiers."""
__all__ = [
    "CNNClassifier",
    "FCNClassifier",
    "LSTMFCNClassifier",
    "MLPClassifier",
    "TapNetClassifier",
]

from aeon.classification.deep_learning.cnn import CNNClassifier
from aeon.classification.deep_learning.fcn import FCNClassifier
from aeon.classification.deep_learning.lstmfcn import LSTMFCNClassifier
from aeon.classification.deep_learning.mlp import MLPClassifier
from aeon.classification.deep_learning.tapnet import TapNetClassifier
