"""Deep learning based regressors."""

__all__ = [
    "CNNRegressor",
    "FCNRegressor",
    "InceptionTimeRegressor",
    "IndividualInceptionRegressor",
    "ResNetRegressor",
    "TapNetRegressor",
    "IndividualLITERegressor",
    "LITETimeRegressor",
    "EncoderRegressor",
    "MLPRegressor",
]

from aeon.regression.deep_learning._cnn import CNNRegressor
from aeon.regression.deep_learning._encoder import EncoderRegressor
from aeon.regression.deep_learning._fcn import FCNRegressor
from aeon.regression.deep_learning._inception_time import (
    InceptionTimeRegressor,
    IndividualInceptionRegressor,
)
from aeon.regression.deep_learning._lite_time import (
    IndividualLITERegressor,
    LITETimeRegressor,
)
from aeon.regression.deep_learning._mlp import MLPRegressor
from aeon.regression.deep_learning._resnet import ResNetRegressor
from aeon.regression.deep_learning._tapnet import TapNetRegressor
