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
from aeon.classification.deep_learning._cnn import CNNClassifier
from aeon.classification.deep_learning._encoder import EncoderClassifier
from aeon.classification.deep_learning._fcn import FCNClassifier
from aeon.classification.deep_learning._inception_time import (
    InceptionTimeClassifier,
    IndividualInceptionClassifier,
)
from aeon.classification.deep_learning._mlp import MLPClassifier
from aeon.classification.deep_learning._resnet import ResNetClassifier
from aeon.classification.deep_learning._tapnet import TapNetClassifier
from aeon.classification.deep_learning.base import BaseDeepClassifier
