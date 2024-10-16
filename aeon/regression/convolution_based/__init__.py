"""Convolution-based time series extrinsic regressors."""

__all__ = [
    "RocketRegressor",
    "MiniRocketRegressor",
    "MultiRocketRegressor",
    "HydraRegressor",
    "MultiRocketHydraRegressor",
]

from aeon.regression.convolution_based._hydra import HydraRegressor
from aeon.regression.convolution_based._minirocket import MiniRocketRegressor
from aeon.regression.convolution_based._mr_hydra import MultiRocketHydraRegressor
from aeon.regression.convolution_based._multirocket import MultiRocketRegressor
from aeon.regression.convolution_based._rocket import RocketRegressor
