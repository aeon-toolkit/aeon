"""Deep learning based regressors."""

__all__ = [
    "CNNRegressor",
    "FCNRegressor",
    "InceptionTimeRegressor",
    "IndividualInceptionRegressor",
    "ResNetRegressor",
    "TapNetRegressor",
]

from aeon.regression.deep_learning._cnn import CNNRegressor
from aeon.regression.deep_learning._fcn import FCNRegressor
from aeon.regression.deep_learning._inception_time import (
    InceptionTimeRegressor,
    IndividualInceptionRegressor,
)
from aeon.regression.deep_learning._resnet import ResNetRegressor
from aeon.regression.deep_learning._tapnet import TapNetRegressor
