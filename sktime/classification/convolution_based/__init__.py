# -*- coding: utf-8 -*-
"""Convolution based time series classifiers."""

__all__ = ["RocketClassifier", "Arsenal", "TimeSeriesSVC"]

from sktime.classification.convolution_based._arsenal import Arsenal
from sktime.classification.convolution_based._rocket_classifier import RocketClassifier
from sktime.classification.kernel_based._svc import TimeSeriesSVC
