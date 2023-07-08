# -*- coding: utf-8 -*-
"""Deep learning based classifiers."""
__all__ = [
    "CNNClassifier",
    "FCNClassifier",
    "MLPClassifier",
    "TapNetClassifier",
    "InceptionTimeClassifier",
]

from aeon.classification.deep_learning.cnn import CNNClassifier
from aeon.classification.deep_learning.fcn import FCNClassifier
from aeon.classification.deep_learning.inception_time import InceptionTimeClassifier
from aeon.classification.deep_learning.mlp import MLPClassifier
from aeon.classification.deep_learning.tapnet import TapNetClassifier
