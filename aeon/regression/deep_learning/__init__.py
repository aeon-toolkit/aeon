# -*- coding: utf-8 -*-
"""Deep learning based regressors."""
__all__ = [
    "CNNRegressor",
    "FCNRegressor",
    "InceptionTimeRegressor",
    "IndividualInceptionRegressor",
    "ResNetRegressor",
    "TapNetRegressor",
]

from aeon.regression.deep_learning.cnn import CNNRegressor
from aeon.regression.deep_learning.fcn import FCNRegressor
from aeon.regression.deep_learning.inception_time import (
    InceptionTimeRegressor,
    IndividualInceptionRegressor,
)
from aeon.regression.deep_learning.resnet import ResNetRegressor
from aeon.regression.deep_learning.tapnet import TapNetRegressor
