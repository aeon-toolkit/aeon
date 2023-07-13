# -*- coding: utf-8 -*-
"""Deep learning based classifiers."""
__all__ = [
    "BaseDeepClassifier",
    "CNNClassifier",
    "EncoderClassifier",
    "FCNClassifier",
    "InceptionTimeClassifier",
    "IndividualInceptionClassifier",
    "MLPClassifier",
    "ResNetClassifier",
    "TapNetClassifier",
]
from aeon.classification.deep_learning.base import BaseDeepClassifier
from aeon.classification.deep_learning.cnn import CNNClassifier
from aeon.classification.deep_learning.encoder import EncoderClassifier
from aeon.classification.deep_learning.fcn import FCNClassifier
from aeon.classification.deep_learning.inception_time import (
    InceptionTimeClassifier,
    IndividualInceptionClassifier,
)
from aeon.classification.deep_learning.mlp import MLPClassifier
from aeon.classification.deep_learning.resnet import ResNetClassifier
from aeon.classification.deep_learning.tapnet import TapNetClassifier
