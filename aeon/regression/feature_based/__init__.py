# -*- coding: utf-8 -*-
"""Feature based time series regressors.

While a bit vague, the contents mostly consist of transformers that extract features
pipelined to a vector regressor.
"""

__all__ = [
    "FreshPRINCERegressor",
]

from aeon.regression.feature_based._fresh_prince import FreshPRINCERegressor
