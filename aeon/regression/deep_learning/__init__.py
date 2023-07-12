# -*- coding: utf-8 -*-
"""Deep learning based regressors."""
__all__ = [
    "CNNRegressor",
    "InceptionTimeRegressor",
    "IndividualInceptionRegressor",
    "TapNetRegressor",
]

from aeon.regression.deep_learning.cnn import CNNRegressor
from aeon.regression.deep_learning.inception_time import (
    InceptionTimeRegressor,
    IndividualInceptionRegressor,
)
from aeon.regression.deep_learning.tapnet import TapNetRegressor
